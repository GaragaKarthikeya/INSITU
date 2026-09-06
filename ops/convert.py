"""The one place floating point becomes fixed point.

Everything upstream (projection, and the weights themselves) is fp16/fp32.
Everything downstream (rotation, quantisation, the cache, attention) is
integer. The boundary is here, in one function, and nowhere else.

Keeping it to one call site is what makes the conversion auditable. A cast
added wherever it happened to be convenient is the standard way two
implementations of the same arithmetic come to disagree on inputs nobody
tested, and it is invisible in review because each individual cast looks
harmless.

The conversion saturates rather than wrapping, and reports how often it had
to. A saturating Q/K/V element is a scaling problem in the weights; silently
wrapping it produces a large-magnitude value of the wrong sign, which softmax
then turns into a confidently wrong attention distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import Q


@dataclass
class ConversionStats:
    n_values: int = 0
    n_saturated: int = 0
    max_abs_float: float = 0.0

    @property
    def saturation_rate(self) -> float:
        return self.n_saturated / self.n_values if self.n_values else 0.0

    def merge(self, other: "ConversionStats") -> None:
        self.n_values += other.n_values
        self.n_saturated += other.n_saturated
        self.max_abs_float = max(self.max_abs_float, other.max_abs_float)


def to_fixed(x: np.ndarray, q: Q, stats: ConversionStats | None = None) -> np.ndarray:
    """fp32 -> Q(frac) integer codes, rounding half to even, saturating.

    Half-to-even and not half-up: this runs on every element of every Q, K and
    V in the model, so a rule biased in one direction is a systematic offset on
    the whole activation tensor, not a rounding error on one value.
    """
    a = np.asarray(x, dtype=np.float64)
    scaled = np.rint(a * q.scale).astype(np.int64)
    if stats is not None:
        stats.n_values += a.size
        stats.n_saturated += q.overflow_count(scaled)
        stats.max_abs_float = max(stats.max_abs_float, float(np.abs(a).max(initial=0.0)))
    return q.clamp(scaled)


def to_float(x: np.ndarray, q: Q) -> np.ndarray:
    """Fixed -> float. The return leg, used at the block's output boundary."""
    return q.to_float(x)
