"""The output projection, and the offline fold that removes the inverse rotation.

Why there is no post-rotation stage
-----------------------------------
The value accumulator comes out of attention in the rotated domain. The naive
fix is to rotate it back per step. The fold does it for free instead, once,
when the weights are loaded:

`Rotation.apply` acts on a row vector as `acc = v @ R.T`, so recovering the
head-space value is `v = acc @ R` (using `R^-1 == R.T`, twice). Then:

    y = v @ W_o.T
      = concat_h(acc_h @ R) @ W_o.T
      = concat_h(acc_h) @ blockdiag_h(R) @ W_o.T
      = concat_h(acc_h) @ (W_o @ blockdiag_h(R).T).T
      = linear(concat_h(acc_h), W_o')      with  W_o' = W_o @ blockdiag_h(R).T

`W_o'` has exactly the shape of `W_o` -- the fold costs no memory and no
runtime.

The transpose on the block is the whole content of this derivation, and it is
easy
to get wrong: `blockdiag(R)` and `blockdiag(R).T` are both orthonormal, both
produce output of the right shape, and both give a model that runs. Only one
gives the right answer. `tests/test_kernel.py` checks the fold against an
explicit rotate-then-project rather than against itself.

Two consequences that must travel with any result
-------------------------------------------------
1. The block's output is meaningless against an unfolded `W_o`. Not slightly
   wrong -- it is in a different basis. `fold_o_proj` is not an optimisation
   that can be skipped.
2. The concatenation order of the accumulators is part of the format. The fold
   builds `blockdiag` head-major, so `merge_heads` must lay the heads out
   head-major too. Getting this wrong produces fluent nonsense rather than an
   error.
"""

from __future__ import annotations

import numpy as np

from ..config import ModelConfig
from ..ops.rotate import Rotation
from ..trace import Trace
from .project import project


def fold_o_proj(o_proj: np.ndarray, rotation: Rotation, model: ModelConfig) -> np.ndarray:
    """`W_o' = W_o @ blockdiag_h(R)`, computed once at weight-load time.

    `o_proj` is (hidden_size, num_heads * head_dim), the checkpoint's layout.
    The fold is exact in real arithmetic -- it only re-associates a linear map
    -- so the only question it raises is the precision it is stored at. It is
    computed in float64 and returned in the input dtype, which for an fp16
    checkpoint means one rounding, at load time, on a weight rather than on an
    activation.
    """
    w = np.asarray(o_proj)
    expected = model.num_heads * model.head_dim
    if w.shape[1] != expected:
        raise ValueError(
            f"o_proj takes {w.shape[1]} input channels but the block produces "
            f"{expected} ({model.num_heads} heads x {model.head_dim}); the fold "
            f"would be built against the wrong layout"
        )
    if rotation.d != model.head_dim:
        raise ValueError(f"rotation is {rotation.d}-dimensional, head_dim is {model.head_dim}")

    r = rotation.matrix().T                                # (d, d); R^-1 on a row
    # Head-major block diagonal. Built explicitly rather than with a reshape
    # trick so the layout assumption is visible at the point it is made.
    block = np.zeros((expected, expected), dtype=np.float64)
    for h in range(model.num_heads):
        s = h * model.head_dim
        block[s:s + model.head_dim, s:s + model.head_dim] = r

    return (w.astype(np.float64) @ block).astype(w.dtype)


def output_project(w_o_folded: np.ndarray, acc: np.ndarray, *,
                   chunk: int | None = None, trace: Trace | None = None) -> np.ndarray:
    """Apply the folded output projection. `acc` is head-major, (…, heads*d)."""
    return project(w_o_folded, acc, unit="o_array", chunk=chunk, trace=trace)
