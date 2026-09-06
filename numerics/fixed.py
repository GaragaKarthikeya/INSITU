"""Fixed-point arithmetic. This is the only module that narrows a value.

Everywhere else, intermediate results stay in int64 and are allowed to grow. A
value only gets narrower through one of the named calls here: `Q.rshift`,
`Q.clamp` or `Q.from_float`. There are no shifts or masks anywhere else in the
package.

The reason is that where you truncate is what defines a fixed-point design. A
stray `>> 3` added because it was convenient is the easiest way for the model
and the hardware to drift apart, and they will still agree on whatever inputs
you happened to test.

Rounding is a parameter, not a default
--------------------------------------
There are three modes, because the difference is real and each has a place:

    "floor"  arithmetic shift right. Cheapest, biased towards -inf.
    "even"   round half to even. Unbiased. This is what a norm conversion
             wants: norms are always positive, so floor would bias every one
             of them the same way and the error would build up across a whole
             cache instead of averaging out.
    "up"     round half away from zero. One adder. Fine on a cold path.

Python's `>>` on a negative int floors, which is what an arithmetic shift right
does, so "floor" is the plain operator and the other two are built on top of
it.
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

    `n` can be an array, broadcast against `x`. There is one caller that needs
    this: the reciprocal in `_finalize` is normalised per row, so two query
    heads with different denominators get shifted by different amounts in the
    same step.

    A scalar `n` still takes the path it always did. The array case is a branch
    inside the same function rather than a second copy of it, so the scalar
    behaviour cannot drift away from it.
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
        # `1 << (n - 1)` gets evaluated for every element, including n == 0,
        # where the shift is -1 and numpy's answer is undefined. The rounding
        # modes below need that half, so we compute it with a floor of 1 and
        # then select it away afterwards. An early return would be simpler but
        # an array cannot take one.
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
    """`max(|x|)` without building `|x|` first.

    `np.abs(x).max()` allocates a temporary the size of `x`. The arrays this
    package reduces over are tens of millions of elements and get reduced
    several times per tile, and there the allocation plus the extra pass over
    memory costs more than the reduction does. Two reductions over the original
    array beat one copy plus one reduction over it.
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
        """Float in, integer code out. The only float to fixed conversion there is.

        Rounds half to even by default. This runs on every Q, K and V element
        of every token, so a biased rule would not be a rounding error on one
        value -- it would be a systematic offset on the entire cache.
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
        """`(clamped, overflow_count, absmax)`, in one pass when nothing overflows.

        Counting overflows and clamping separately takes four passes over the
        array. On the hot path nothing overflows at all, and a single `absmax`
        is enough to prove that. If something does overflow we fall through to
        the slow path, so no count is ever approximated -- we just do not pay
        for counting when there is nothing to count.
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
    """Unnormalised Walsh-Hadamard transform. Adds and subtracts, nothing else.

    Being exact on integers is the whole reason the rotation is a Hadamard and
    not some arbitrary orthogonal matrix: there is no truncation inside it to
    account for, at any width.

    Works on the last axis, in place on a copy, and runs the stages in the same
    order the butterfly does -- so this and the hardware agree on the
    intermediates too, not only on the final value.
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
    """`log2(sqrt(n))` if that is a whole number, otherwise None.

    `1/sqrt(n)` is an exact right shift for even powers of two and for nothing
    else. A caller that gets None has to use a Q15 reciprocal multiply instead.
    Shifting by a rounded exponent would give a wrong answer that looks
    perfectly plausible, which is why this returns None rather than guessing.
    """
    if n & (n - 1) or n < 1:
        return None
    e = n.bit_length() - 1
    return e // 2 if e % 2 == 0 else None


def inv_sqrt_q15(n: int) -> int:
    """round(2^15 / sqrt(n)), for the odd-power case `sqrt_shift` rejects."""
    import math
    return int(round((1 << 15) / math.sqrt(n)))
