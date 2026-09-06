"""Rotary position embedding, applied where the hardware would apply it.

Position in the pipeline is not negotiable
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

Frequency scaling
-----------------
Llama 3 rescales `inv_freq` by wavelength before the table is built (
`RopeScaling`, below). It is a change to the table and nothing else: the
`apply` path, the trace it records and the cycles the hardware model replays
are identical either way. That is the only reason it belongs here rather than
in the pipeline -- a position-dependent transform that cost cycles could not be
folded into a precomputed table.

Fixed point
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
class RopeScaling:
    """Llama 3's wavelength-piecewise rescaling of `inv_freq`.

    Three regimes, split by how a channel's wavelength compares to the context
    the model was originally trained on:

      * **short** (`wavelen < ctx / high_freq_factor`) -- untouched. These
        channels turn over many times inside the old context, so stretching
        them would destroy local position information.
      * **long** (`wavelen > ctx / low_freq_factor`) -- divided by `factor`.
        These never complete a turn, so slowing them is what buys the longer
        context.
      * **between** -- linearly interpolated between the two, which is the
        whole point of the scheme: a hard switch would put a discontinuity in
        the middle of the frequency band.

    It is not a position-dependent transform. Every channel's frequency is
    fixed once, so this is a change to the table alone -- see the module
    docstring.

    Matches `transformers.modeling_rope_utils._compute_llama3_parameters`; the
    `attention_factor` that function also returns is 1.0 for every llama3
    config, and is not applied here.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position: int

    def __post_init__(self) -> None:
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError(
                f"high_freq_factor={self.high_freq_factor} must exceed "
                f"low_freq_factor={self.low_freq_factor}; the interpolation "
                f"between them divides by the difference"
            )
        if self.factor <= 0:
            raise ValueError(f"factor={self.factor} must be positive")

    @classmethod
    def from_hf(cls, params: dict) -> "RopeScaling":
        """Build from a transformers `rope_parameters` / `rope_scaling` dict."""
        return cls(
            factor=float(params["factor"]),
            low_freq_factor=float(params["low_freq_factor"]),
            high_freq_factor=float(params["high_freq_factor"]),
            original_max_position=int(params["original_max_position_embeddings"]),
        )

    def apply(self, inv_freq: np.ndarray) -> np.ndarray:
        f = np.asarray(inv_freq, dtype=np.float64)
        ctx = float(self.original_max_position)
        wavelen = 2.0 * np.pi / f

        low_wavelen = ctx / self.low_freq_factor
        high_wavelen = ctx / self.high_freq_factor

        scaled = np.where(wavelen > low_wavelen, f / self.factor, f)

        smooth = ((ctx / wavelen - self.low_freq_factor) /
                  (self.high_freq_factor - self.low_freq_factor))
        smoothed = (1.0 - smooth) * scaled / self.factor + smooth * scaled

        between = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
        return np.where(between, smoothed, scaled)


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
    def build(d: int, max_position: int, theta: float = 10000.0, frac: int = 15,
              scaling: "RopeScaling | None" = None) -> "RoPE":
        if d % 2:
            raise ValueError(f"head_dim={d} must be even: RoPE rotates channel pairs")
        half = d // 2
        inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / d))
        if scaling is not None:
            inv_freq = scaling.apply(inv_freq)
        pos = np.arange(max_position, dtype=np.float64)[:, None]
        ang = pos * inv_freq[None, :]
        q = Q(32, frac)
        return RoPE(cos=q.from_float(np.cos(ang)), sin=q.from_float(np.sin(ang)), frac=frac)

    def apply(self, x, positions, qk_frac: int,
              trace: Trace | None = None, unit: str = "") -> np.ndarray:
        """Rotate `x` (..., d) in Q(qk_frac) by the angle for each position.

        Channel pairing is the half-split `(x[:half], x[half:])` that Llama
        uses, not the interleaved `(x[0::2], x[1::2])` of the original paper.
        The two are related by a fixed permutation and are not interchangeable
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
