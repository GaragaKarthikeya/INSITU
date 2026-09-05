"""Fixed-point arithmetic: the one module allowed to narrow a value.

THE DISCIPLINE
--------------
Intermediate arithmetic is int64 and grows naturally. A value narrows ONLY at a
named call here -- `Q.rshift`, `Q.clamp`, `Q.from_float`. Nothing elsewhere in
the package writes a shift or a mask.

The reason is that a fixed-point design's identity IS its truncation points. An
ad hoc `>> 3` added somewhere convenient is the most likely way two
implementations of the same arithmetic diverge on some inputs while still
agreeing on the ones anyone tested.

ROUNDING IS A PARAMETER, NOT A DEFAULT
--------------------------------------
Three modes, because the difference is real and each has a right place:

    "floor"  arithmetic shift right. Cheapest; biased toward -inf.
    "even"   round half to even. Unbiased. What a norm conversion wants,
             because norms are always positive and floor's bias accumulates
             across an entire cache.
    "up"     round half away from zero. One adder. Fine on a cold path.

Python's `>>` on a negative int floors, which matches an arithmetic shift
right, so "floor" is the operator and the other two are built from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT = np.int64
Rounding = str  # "floor" | "even" | "up"


def _as_int(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind not in "iu":
        raise TypeError(
            f"fixed-point op on dtype {a.dtype}; convert with Q.from_float first "
            f"so the conversion point is visible"
        )
    return a.astype(INT, copy=False)


def rshift(x, n, rounding: Rounding = "floor") -> np.ndarray:
    """`x >> n` under the named rounding rule. `n == 0` is a no-op, not an error.

    `n` may be an ARRAY, broadcast against `x`. That exists for one caller:
    `_finalize`'s reciprocal is normalised per row, so two query heads with
    different denominators are shifted by different amounts in the same step.
    A scalar `n` takes the same path it always did -- the array case is a
    branch, not a reimplementation, so the scalar behaviour cannot drift.
    """
    a = _as_int(x)
    if np.ndim(n) == 0:
        if n < 0:
            raise ValueError(f"rshift by {n}: use lshift for a left shift, so direction is explicit")
        if n == 0:
            return a
    else:
        n = _as_int(n)
        if np.any(n < 0):
            raise ValueError("rshift by a negative amount: use lshift, so direction is explicit")
        # `1 << (n - 1)` is evaluated for every element including n == 0, where
        # the shift is -1 and numpy's answer is undefined. The rounding modes
        # below need the half, so n == 0 is handled by selecting it away rather
        # than by an early return that an array cannot take.
        safe = np.maximum(n, INT(1))
    if rounding == "floor":
        return a >> n
    half = (INT(1) << (safe - 1)) if np.ndim(n) else (INT(1) << (n - 1))
    if rounding == "up":
        q = (a + half) >> n
        return np.where(n == 0, a, q) if np.ndim(n) else q
    if rounding == "even":
        q = (a + half) >> n
        # A tie is exactly representable: the discarded bits are 100...0.
        tie = (a & ((INT(1) << n) - 1)) == half
        q = np.where(tie, q & ~INT(1), q)
        return np.where(n == 0, a, q) if np.ndim(n) else q
    raise ValueError(f"unknown rounding {rounding!r}; expected floor, even or up")


def lshift(x, n: int) -> np.ndarray:
    return _as_int(x) << n


def absmax(x) -> int:
    """`max(|x|)` without allocating `|x|`.

    `np.abs(x).max()` materialises a full temporary the size of `x`. On the
    arrays this package reduces over -- tens of millions of elements, several
    times per tile -- that allocation and the extra memory pass cost more than
    the reduction itself. Two reductions over the original beat one pass plus
    one reduction over a copy.
    """
    a = np.asarray(x)
    if a.size == 0:
        return 0
    return int(max(a.max(), -a.min()))


@dataclass(frozen=True)
class Q:
    """A signed two's-complement fixed-point format: `width` bits, `frac` of them fractional."""

    width: int
    frac: int
    saturate: bool = True

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError(f"width={self.width} leaves no room for a sign bit and a magnitude")
        if not 0 <= self.frac < self.width:
            raise ValueError(f"frac={self.frac} outside [0, {self.width})")

    @property
    def scale(self) -> int:
        return 1 << self.frac

    @property
    def lo(self) -> int:
        return -(1 << (self.width - 1))

    @property
    def hi(self) -> int:
        return (1 << (self.width - 1)) - 1

    def from_float(self, x, rounding: Rounding = "even") -> np.ndarray:
        """Float in, integer code out. The ONLY float -> fixed conversion point.

        Rounds half to even by default: this runs on every Q/K/V element of
        every token, so a biased rule would be a systematic offset on the
        whole cache rather than a rounding error on one value.
        """
        a = np.asarray(x, dtype=np.float64) * self.scale
        if rounding == "even":
            r = np.rint(a)              # numpy's rint is round-half-to-even
        elif rounding == "up":
            r = np.floor(a + 0.5)
        elif rounding == "floor":
            r = np.floor(a)
        else:
            raise ValueError(f"unknown rounding {rounding!r}")
        return self.clamp(r.astype(INT))

    def to_float(self, x) -> np.ndarray:
        return _as_int(x).astype(np.float64) / self.scale

    def clamp(self, x) -> np.ndarray:
        a = _as_int(x)
        if not self.saturate:
            # Wrap, which is what a register does when nothing checks it.
            span = INT(1) << self.width
            return ((a + (INT(1) << (self.width - 1))) % span) - (INT(1) << (self.width - 1))
        return np.clip(a, self.lo, self.hi)

    def fits(self, x) -> np.ndarray:
        a = _as_int(x)
        return (a >= self.lo) & (a <= self.hi)

    def overflow_count(self, x) -> int:
        """How many elements would saturate. Reported, never silently absorbed."""
        return int((~self.fits(x)).sum())

    def check_and_clamp(self, x) -> tuple[np.ndarray, int, int]:
        """`(clamped, overflow_count, absmax)` in one reduction on the common path.

        Counting overflows and clamping separately is four passes over the
        array; on the hot path nothing overflows, and a single `absmax` proves
        it. When it does overflow the slow path runs, so nothing is
        approximated -- the saturation count stays exact, it is just not paid
        for when there is nothing to count.
        """
        a = _as_int(x)
        peak = absmax(a)
        if peak <= self.hi and -peak >= self.lo:
            return a, 0, peak
        return self.clamp(a), self.overflow_count(a), min(peak, max(self.hi, -self.lo))

    def requantize(self, x, src_frac: int, rounding: Rounding = "floor") -> np.ndarray:
        """Move a value from Q(src_frac) into this format, then clamp."""
        shift = src_frac - self.frac
        if shift > 0:
            x = rshift(x, shift, rounding)
        elif shift < 0:
            x = lshift(x, -shift)
        return self.clamp(x)

    def __repr__(self) -> str:
        return f"Q{self.width - self.frac}.{self.frac}"


def fwht(x) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform: adds and subtracts only.

    Exact on integers, which is the entire reason the rotation is a Hadamard
    and not an arbitrary orthogonal matrix -- there is no truncation inside it
    to account for, at any width.

    Operates on the last axis, in place on a copy, in the butterfly's own
    stage order so the hardware and this agree on more than the final value.
    """
    a = _as_int(x).copy()
    n = a.shape[-1]
    if n & (n - 1):
        raise ValueError(f"fwht length {n} is not a power of two")
    h = 1
    while h < n:
        a = a.reshape(*a.shape[:-1], n // (2 * h), 2, h)
        lo = a[..., 0, :].copy()
        hi = a[..., 1, :].copy()
        a[..., 0, :] = lo + hi
        a[..., 1, :] = lo - hi
        a = a.reshape(*a.shape[:-3], n)
        h *= 2
    return a


def sqrt_shift(n: int) -> int | None:
    """`log2(sqrt(n))` when it is an integer, else None.

    1/sqrt(n) is an exact right shift for even powers of two, and only then.
    Callers that get None must use a Q15 reciprocal multiply instead; silently
    shifting by a rounded exponent is a wrong answer that looks plausible.
    """
    if n & (n - 1) or n < 1:
        return None
    e = n.bit_length() - 1
    return e // 2 if e % 2 == 0 else None


def inv_sqrt_q15(n: int) -> int:
    """round(2^15 / sqrt(n)), for the odd-power case `sqrt_shift` rejects."""
    import math
    return int(round((1 << 15) / math.sqrt(n)))
