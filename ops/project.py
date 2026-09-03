"""The Q/K/V and output projections.

Values only. The systolic array's *schedule* lives in `kernel.hw.pe_array` and
reads the trace this emits; nothing here knows how many PEs there are.

The one thing that does leak through is the reduction chunk, because the order
of a floating-point sum changes its value. `chunk` is the array's row count
when a hardware config is present and `None` (numpy's own order) when it is
not, so a functional-mode run and a hardware-mode run of the same kernel differ
in the last bit or two of the projection. That is a real property of the
machine, not a modelling defect, and it is stated rather than smoothed over.
"""

from __future__ import annotations

import numpy as np

from ..numerics import fp
from ..trace import Op, Trace, record


def project(w: np.ndarray, x: np.ndarray, *, unit: str,
            bias: np.ndarray | None = None,
            chunk: int | None = None, trace: Trace | None = None) -> np.ndarray:
    """`x @ w.T` in fp16 with fp32 accumulation.

    `w` is (out_features, in_features) -- the checkpoint's own layout, kept so
    a weight tensor can be used without a transpose or a copy.
    `x` is (in_features,) or (tokens, in_features).
    """
    w = np.asarray(w)
    x = np.asarray(x)
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got {w.shape}")
    if x.shape[-1] != w.shape[1]:
        raise ValueError(f"cannot project {x.shape} through {w.shape}")

    if chunk is None:
        out = fp.matvec(w, x)
    elif x.ndim == 1:
        out = fp.matvec_ordered(w, x, chunk)
    else:
        out = np.stack([fp.matvec_ordered(w, row, chunk) for row in x])

    if bias is not None:
        # Added after the reduction, in fp32, exactly as a linear layer does.
        # Qwen2.5 has QKV biases and Llama does not, so this must be optional
        # rather than assumed absent -- a dropped bias is a silent wrongness.
        out = out + np.asarray(bias, dtype=fp.ACC)

    n_tokens = 1 if x.ndim == 1 else x.shape[0]
    record(trace, Op.MATVEC, unit,
           m=w.shape[0], k=w.shape[1], n=n_tokens,
           n_bytes=w.size * 2)     # fp16 weights, streamed once per call
    return out


def split_heads(x: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """(tokens, n_heads*head_dim) -> (tokens, n_heads, head_dim)."""
    x = np.asarray(x)
    lead = x.shape[:-1]
    if x.shape[-1] != n_heads * head_dim:
        raise ValueError(f"cannot split {x.shape[-1]} into {n_heads}x{head_dim}")
    return x.reshape(*lead, n_heads, head_dim)


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(tokens, n_heads, head_dim) -> (tokens, n_heads*head_dim).

    Head-major, which is the order `W_o` expects and therefore the order the
    per-head rotations must be block-diagonalised in when `W_o` is folded.
    """
    x = np.asarray(x)
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
