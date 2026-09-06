"""Scalar codebooks, computed rather than tabulated.

Why there is no calibration file
--------------------------------
Rotating a vector by a randomized Walsh-Hadamard transform makes its channels
close to i.i.d. Gaussian, which is the whole reason the rotation is there. So
the best scalar codebook to use afterwards is the Lloyd-Max quantizer for
N(0,1), and that depends on the bit-width and nothing else.

That removes a whole class of infrastructure: no calibration pass over model
activations, no exported ROM, no JSON that the hardware and the golden model
have to be kept in sync on, and no way to write a cache under one calibration
and read it under another. `Codebook.gaussian(bits)` is a pure function of an
integer and gives the same bits on any machine.

The claim that rotated channels end up Gaussian is checked rather than
assumed. `tests/test_rotation.py` measures it two ways:
`check_gaussianises_outlier_heavy_activations` on activations with outlier
channels, and `check_one_round_overshoots_into_platykurtic` on the
pathological one-hot case.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import INT, Q


@dataclass(frozen=True)
class Codebook:
    """`2**bits` reconstruction levels and the `2**bits - 1` decision boundaries.

    Both are stored as integers in Q(frac). The encoder compares against the
    boundaries; the decoder multiplies by the centroids. They are two separate
    arrays rather than one derived from the other on use, because the midpoint
    rule relating them holds for Lloyd-Max but would not hold for every
    codebook someone might plug in later.
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

        Solved on a fixed grid rather than in closed form. Lloyd's algorithm on
        a 65536-point discretisation of N(0,1) lands well within the Q1.15
        quantum we store the answer at, so the grid is invisible in the output.
        Cached, so a given width costs this once per process.

        `symmetric` forces `c[i] == -c[n-1-i]` exactly. Lloyd's optimum for a
        symmetric density already comes out symmetric to within solver noise;
        forcing it makes that exact, which lets the decoder store a magnitude
        table and a sign bit instead of the full table. It costs far less than
        one quantum of distortion.
        """
        if bits < 1:
            raise ValueError(f"bits={bits} must be at least 1")
        n = 1 << bits

        # A symmetric grid carrying N(0,1) mass. +-8 sigma covers all but
        # about 1e-15 of it.
        x = np.linspace(-8.0, 8.0, grid)
        w = np.exp(-0.5 * x * x)
        w /= w.sum()

        # Start from equiprobable quantiles. That is closer to the optimum than
        # a uniform split, so Lloyd converges without hunting around.
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
        # Midpoints rounded down, so a value sitting exactly on a boundary
        # falls into the lower bin. The encoder in `quantize` uses `>` for the
        # same reason. The two have to agree or every tie shifts by one code.
        bq = (cq[:-1] + cq[1:]) >> INT(1)
        bq = _force_strictly_ascending(bq)
        return Codebook(centroids=cq, boundaries=bq, bits=bits, frac=frac)


    @staticmethod
    @functools.lru_cache(maxsize=64)
    def uniform(bits: int, frac: int = 15, symmetric: bool = True,
                grid: int = 1 << 16) -> "Codebook":
        """MSE-optimal equally-spaced quantizer for the standard normal.

        This is the control for `gaussian`. Equal spacing is cheaper in
        hardware -- the decision boundaries become a multiply and a shift
        instead of a comparator tree -- so if it turns out to cost nothing in
        accuracy, then the Lloyd-Max codebook is not earning its keep and the
        "no calibration needed" argument is really about the rotation rather
        than about the codebook.

        The only thing optimised here is the clipping range, found by scanning.
        Everything else about a uniform quantizer is fixed by construction.
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
    """Break ties that rounding to Q(frac) introduced.

    At small `bits` and large `frac` this never fires. Once `bits` gets close to
    the format's resolution, two adjacent levels can round to the same integer,
    and a boundary array that is not strictly increasing quietly breaks
    `searchsorted`.

    Nudging is the right fix here because the levels really are distinct before
    rounding. What we are repairing is the storage format, not the solution.
    """
    a = a.astype(INT).copy()
    for i in range(1, len(a)):
        if a[i] <= a[i - 1]:
            a[i] = a[i - 1] + 1
    return a
