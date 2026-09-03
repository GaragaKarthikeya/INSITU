"""Rotary position embedding, applied where the hardware would apply it.

POSITION IN THE PIPELINE IS NOT NEGOTIABLE
------------------------------------------
RoPE sits between the projection and the Hadamard rotation:

    x -> W_qkv -> RoPE -> R -> quantize -> cache

It must be before `R`, because what gets cached is the post-RoPE key: RoPE is
position-dependent, so a cache of pre-RoPE keys would have to be re-rotated at
every future step, which is exactly the O(context) work this design exists to
remove. And it must be before quantisation for the same reason.

It is also why `R` and RoPE cannot be merged: RoPE's rotation angle depends on
the token's position, `R`'s sign diagonal does not, and folding a
position-dependent transform into the offline `W_o` fold is not possible.

There is deliberately NO RoPE between the attention output and the output
projection. The value path carries no position, so `W_o'` sees only `R^-1`.

FIXED POINT
-----------
The sin/cos table is computed once in float64 and stored in Q(centroid_frac).
The rotation itself is two multiplies and an add per channel pair, with a
single truncation at the end -- the table is the only approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import INT, Q, rshift
from ..trace import Op, Trace, record


@dataclass(frozen=True)
class RoPE:
    """Precomputed sin/cos for every position the model can reach.

    Built eagerly for `max_position` because it is small -- `max_position * d`
    int64s, 2 MB at 4096x64 -- and because computing it lazily makes the first
    token of a long context slower than the rest for no reason a profile would
    explain.
    """

    cos: np.ndarray      # (max_position, d // 2) int64, Q(frac)
    sin: np.ndarray
    frac: int

    @staticmethod
    def build(d: int, max_position: int, theta: float = 10000.0, frac: int = 15) -> "RoPE":
        if d % 2:
            raise ValueError(f"head_dim={d} must be even: RoPE rotates channel pairs")
        half = d // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / d))
        pos = np.arange(max_position, dtype=np.float64)[:, None]
        ang = pos * inv_freq[None, :]
        q = Q(32, frac)
        return RoPE(cos=q.from_float(np.cos(ang)), sin=q.from_float(np.sin(ang)), frac=frac)

    def apply(self, x, positions, qk_frac: int,
              trace: Trace | None = None, unit: str = "") -> np.ndarray:
        """Rotate `x` (..., d) in Q(qk_frac) by the angle for each position.

        Channel pairing is the half-split `(x[:half], x[half:])` that Llama
        uses, not the interleaved `(x[0::2], x[1::2])` of the original paper.
        The two are related by a fixed permutation and are NOT interchangeable
        against a trained checkpoint -- a model trained under one and served
        under the other produces fluent nonsense rather than an obvious error.
        """
        a = np.asarray(x, dtype=INT)
        d = a.shape[-1]
        half = d // 2
        pos = np.asarray(positions, dtype=INT)
        if pos.max(initial=0) >= self.cos.shape[0]:
            raise ValueError(
                f"position {int(pos.max())} beyond the table's {self.cos.shape[0]}; "
                f"raise ModelConfig.max_position rather than wrapping"
            )

        c = self.cos[pos]                       # (..., half) Q(frac)
        s = self.sin[pos]
        lo, hi = a[..., :half], a[..., half:]

        # Q(qk_frac + frac) throughout, one truncation at the end.
        out_lo = rshift(lo * c - hi * s, self.frac)
        out_hi = rshift(lo * s + hi * c, self.frac)
        out = np.concatenate([out_lo, out_hi], axis=-1)

        record(trace, Op.ROPE, unit, m=d, k=1, n=int(np.prod(a.shape[:-1])) or 1)
        return out

    def apply_float(self, x, positions) -> np.ndarray:
        """float64 reference, same pairing."""
        a = np.asarray(x, dtype=np.float64)
        half = a.shape[-1] // 2
        scale = float(1 << self.frac)
        c = self.cos[np.asarray(positions)] / scale
        s = self.sin[np.asarray(positions)] / scale
        lo, hi = a[..., :half], a[..., half:]
        return np.concatenate([lo * c - hi * s, lo * s + hi * c], axis=-1)
