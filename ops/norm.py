"""Per-head RMS normalisation of the query and key -- "QK-norm".

Why this is in the kernel at all
--------------------------------
It is not optional decoration. Qwen3, OLMo-2 and Gemma-3 all normalise q and k
per head before RoPE, and a model that expects it produces confidently wrong
output without it -- no error, no NaN, just a different distribution. Since the
whole point of this package is to be graftable into a real model, it has to
carry the architecture's actual pre-attention transform.

Where it sits, and why that order
---------------------------------
    project -> split heads -> RMSNorm(q), RMSNorm(k) -> RoPE -> rotate -> cache

Before RoPE, because that is where the reference applies it, and RoPE does not
commute with a per-channel gain. Before the fp -> fixed cast, because it is
float arithmetic in the reference and belongs on the float side of the one
conversion point.

The value path is deliberately not normalised. Only q and k are, because only
they enter the score; V is carried through untouched.

The 1/sqrt(head_dim) attention scale is applied after the norm, not before:
RMSNorm is invariant to the scale of its input, so folding the scale in
earlier would simply be erased.
"""

from __future__ import annotations

import numpy as np

from ..numerics import fp


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """`weight * x / sqrt(mean(x^2) + eps)` over the last axis.

    Computed in float32, matching the reference implementation, which upcasts
    to float32 regardless of the model's dtype. Doing it in float64 instead
    would be "more accurate" and would not be what the model does.
    """
    x = np.asarray(x, dtype=fp.ACC)
    if weight.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"QK-norm gain has {weight.shape[-1]} channels but the head is "
            f"{x.shape[-1]}; this norm is over head_dim only, not hidden_size"
        )
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=fp.ACC)
    return (x * np.reciprocal(np.sqrt(var + fp.ACC(eps)))) * weight.astype(fp.ACC)
