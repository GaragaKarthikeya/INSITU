"""Scalar codebooks, computed rather than tabulated.

WHY THERE IS NO CALIBRATION FILE
--------------------------------
Rotating a vector by a randomized Walsh-Hadamard transform makes its channels
close to i.i.d. Gaussian -- that is the entire reason the rotation is there.
So the optimal scalar codebook after rotation is the Lloyd-Max quantizer for
N(0,1), which depends on the bit-width and nothing else.

That collapses a whole class of infrastructure. No calibration pass over model
activations, no exported ROM, no JSON that hardware and golden model must be
kept in sync on, and no possibility of a cache written under one calibration
being read under another. `Codebook.gaussian(bits)` is a pure function of an
integer, and it is deterministic to the last bit on any machine.

The claim that rotated channels are Gaussian is checkable, not assumed:
`kernel/tests/test_rotation_gaussianity.py` measures the KS distance and the
distortion penalty against a codebook fitted to the real data.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import INT, Q


@dataclass(frozen=True)
class Codebook:
    """`2**bits` reconstruction levels and the `2**bits - 1` decision boundaries.

    Both are stored as integers in Q(frac). Boundaries are what the encoder
    compares against; centroids are what the decoder multiplies by. They are
    kept as separate arrays rather than one derived from the other at use time,
    because the midpoint rule that relates them holds for Lloyd-Max and not for
    every codebook someone might later plug in.
    """

    centroids: np.ndarray      # (2**bits,) int64, Q(frac), ascending
    boundaries: np.ndarray     # (2**bits - 1,) int64, Q(frac), ascending
    bits: int
    frac: int

    def __post_init__(self) -> None:
        n = 1 << self.bits
        if self.centroids.shape != (n,):
            raise ValueError(f"expected {n} centroids, got {self.centroids.shape}")
        if self.boundaries.shape != (n - 1,):
            raise ValueError(f"expected {n - 1} boundaries, got {self.boundaries.shape}")
        if not np.all(np.diff(self.centroids) > 0):
            raise ValueError("centroids must be strictly ascending")
        if not np.all(np.diff(self.boundaries) > 0):
            raise ValueError("boundaries must be strictly ascending")

    @property
    def size(self) -> int:
        return 1 << self.bits

    def to_float(self) -> np.ndarray:
        return self.centroids.astype(np.float64) / (1 << self.frac)

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def gaussian(bits: int, frac: int = 15, symmetric: bool = True,
                 grid: int = 1 << 16, iters: int = 200) -> "Codebook":
        """Lloyd-Max quantizer for the standard normal.

        Solved on a fixed grid rather than in closed form: Lloyd's algorithm on
        a 65536-point discretisation of N(0,1) converges to the continuous
        optimum far below the Q1.15 quantum the result is stored at, so the
        discretisation is invisible in the output. Cached, so the cost is paid
        once per width per process.

        `symmetric` forces `c[i] == -c[n-1-i]` exactly. Lloyd's optimum for a
        symmetric density is symmetric up to solver noise anyway; forcing it
        makes the property exact, which lets the decoder use a magnitude table
        and a sign bit instead of a full one. The distortion cost is far below
        one quantum.
        """
        if bits < 1:
            raise ValueError(f"bits={bits} must be at least 1")
        n = 1 << bits

        # A symmetric grid with N(0,1) mass. +-8 sigma holds all but ~1e-15.
        x = np.linspace(-8.0, 8.0, grid)
        w = np.exp(-0.5 * x * x)
        w /= w.sum()

        # Initialise on equiprobable quantiles: closer to the optimum than a
        # uniform split, so Lloyd converges without hunting.
        cdf = np.cumsum(w)
        c = np.interp((np.arange(n) + 0.5) / n, cdf, x)

        for _ in range(iters):
            b = 0.5 * (c[:-1] + c[1:])
            region = np.searchsorted(b, x)
            mass = np.bincount(region, weights=w, minlength=n)
            moment = np.bincount(region, weights=w * x, minlength=n)
            new = np.where(mass > 0, moment / np.maximum(mass, 1e-300), c)
            if np.allclose(new, c, atol=1e-12):
                c = new
                break
            c = new

        if symmetric:
            c = 0.5 * (c - c[::-1])

        cq = Q(32, frac).from_float(c)
        cq = _force_strictly_ascending(cq)
        # Midpoints, rounded down, so a value exactly on a boundary falls to
        # the lower bin. `searchsorted(..., side="right")` in `quantize` uses
        # the same convention; the two must agree or a code shifts by one.
        bq = (cq[:-1] + cq[1:]) >> INT(1)
        bq = _force_strictly_ascending(bq)
        return Codebook(centroids=cq, boundaries=bq, bits=bits, frac=frac)


    @staticmethod
    @functools.lru_cache(maxsize=64)
    def uniform(bits: int, frac: int = 15, symmetric: bool = True,
                grid: int = 1 << 16) -> "Codebook":
        """MSE-optimal equally-spaced quantizer for the standard normal.

        The control for `gaussian`. Equal spacing is cheaper in hardware -- the
        decision boundaries are a multiply and a shift rather than a comparator
        tree -- so if it costs nothing in accuracy, the Lloyd-Max codebook is
        not earning its place and the "no calibration needed" argument is about
        the ROTATION alone rather than about the codebook.

        Only the clipping range is optimised, by scanning it: everything else
        about a uniform quantizer is fixed by construction.
        """
        n = 1 << bits
        x = np.linspace(-8.0, 8.0, grid)
        w = np.exp(-0.5 * x * x)
        w /= w.sum()

        best, best_mse = None, np.inf
        for c in np.linspace(0.5, 6.0, 400):
            levels = np.linspace(-c, c, n) if n > 1 else np.array([0.0])
            edges = 0.5 * (levels[:-1] + levels[1:])
            mse = float((w * (levels[np.searchsorted(edges, x)] - x) ** 2).sum())
            if mse < best_mse:
                best, best_mse = levels, mse

        cq = _force_strictly_ascending(Q(32, frac).from_float(best))
        bq = _force_strictly_ascending((cq[:-1] + cq[1:]) >> INT(1))
        return Codebook(centroids=cq, boundaries=bq, bits=bits, frac=frac)

    @staticmethod
    def build(kind: str, bits: int, frac: int = 15, symmetric: bool = True) -> "Codebook":
        if kind == "gaussian":
            return Codebook.gaussian(bits, frac, symmetric)
        if kind == "uniform":
            return Codebook.uniform(bits, frac, symmetric)
        raise ValueError(f"unknown codebook {kind!r}")


def _force_strictly_ascending(a: np.ndarray) -> np.ndarray:
    """Break ties introduced by rounding to Q(frac).

    At small `bits` and large `frac` this never fires. At `bits` near the
    format's resolution two adjacent levels can round together, and a
    non-monotone boundary array silently breaks `searchsorted`. Nudging is
    correct here because the levels are genuinely distinct before rounding;
    what is being repaired is the storage format, not the solution.
    """
    a = a.astype(INT).copy()
    for i in range(1, len(a)):
        if a[i] <= a[i - 1]:
            a[i] = a[i - 1] + 1
    return a
