"""TurboQuant as the paper specifies it, in floating point.

    python -m kernel.experiments.turboquant_ref

A reference implementation, written from arXiv:2504.19874 rather than adapted
from this repository, so that the format this project actually stores can be
compared against the published one instead of assumed equal to it.

Where it differs from `ops/quantize.py`, and why that matters
------------------------------------------------------------
    rotation   the paper uses a DENSE random orthogonal matrix from the QR
               decomposition of an i.i.d. Gaussian. This package substitutes a
               randomized Hadamard, which is O(d log d) instead of O(d^2) and
               is what makes the transform affordable in hardware. That
               substitution is an implementation choice, not the paper's.
    norm       the paper stores ||x||_2 in floating point. This package stores
               an RMS norm quantized to Q7.9.
    codebook   both are Lloyd-Max for the normal, but the paper's centroids are
               scaled by 1/sqrt(d), because rotating a unit vector leaves each
               coordinate distributed as N(0, 1/d).
    keys       the paper recommends TurboQuant_prod -- MSE at b-1 bits plus a
               1-bit QJL correction on the residual -- for inner products, and
               TurboQuant_mse for values. Only the MSE variant is implemented
               here; the QJL half is a separate estimator and is not what this
               project stores.

So this is not a drop-in replacement for the kernel's quantizer. It exists to
answer one question: how much of the accuracy gap belongs to the format we
chose, rather than to the arithmetic we run it in.
"""

from __future__ import annotations

import argparse

import numpy as np


def dense_rotation(d: int, seed: int) -> np.ndarray:
    """A random orthogonal matrix, as the paper builds it: QR of a Gaussian.

    Signs of R's diagonal are normalised so the result is Haar-distributed;
    without that, `np.linalg.qr` biases the result.
    """
    g = np.random.default_rng(seed).standard_normal((d, d))
    q, r = np.linalg.qr(g)
    return q * np.sign(np.diag(r))


def lloyd_max_normal(bits: int, grid: int = 1 << 16, iters: int = 300) -> np.ndarray:
    """Lloyd-Max centroids for the standard normal, solved on a fixed grid."""
    n = 1 << bits
    x = np.linspace(-8.0, 8.0, grid)
    w = np.exp(-0.5 * x * x)
    w /= w.sum()
    c = np.interp((np.arange(n) + 0.5) / n, np.cumsum(w), x)
    for _ in range(iters):
        b = 0.5 * (c[:-1] + c[1:])
        region = np.searchsorted(b, x)
        mass = np.bincount(region, weights=w, minlength=n)
        moment = np.bincount(region, weights=w * x, minlength=n)
        new = np.where(mass > 0, moment / np.maximum(mass, 1e-300), c)
        if np.allclose(new, c, atol=1e-13):
            c = new
            break
        c = new
    return 0.5 * (c - c[::-1])          # force exact symmetry


class TurboQuantMSE:
    """Algorithm 1 of the paper. Rotate, then nearest centroid per coordinate."""

    def __init__(self, d: int, bits: int, seed: int = 0):
        self.d = d
        self.bits = bits
        self.R = dense_rotation(d, seed)
        # Coordinates of a rotated unit vector are N(0, 1/d), so the codebook
        # for N(0,1) is scaled down by sqrt(d).
        self.centroids = lloyd_max_normal(bits) / np.sqrt(d)
        self.edges = 0.5 * (self.centroids[:-1] + self.centroids[1:])

    def quantize(self, x):
        """(..., d) real -> (codes, norm). `norm` is the L2 norm, in float."""
        x = np.asarray(x, dtype=np.float64)
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        u = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
        y = u @ self.R.T                                  # rotate
        return np.searchsorted(self.edges, y), norm[..., 0]

    def dequantize(self, codes, norm):
        """Codes and norm -> the reconstruction, in the ORIGINAL basis."""
        y_hat = self.centroids[codes]
        return (y_hat @ self.R) * np.asarray(norm)[..., None]   # R^T, then scale

    def score(self, q, codes, norm):
        """The paper's inner product: reconstruct first, then dot."""
        return np.asarray(q, dtype=np.float64) @ self.dequantize(codes, norm).T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--vectors", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    print("centroids, against the values printed in the paper")
    print("  b=1  paper  +-sqrt(2/pi)/sqrt(d) = +-0.7979/sqrt(d)")
    c1 = lloyd_max_normal(1)
    print(f"       ours   {c1[1]:+.4f}/sqrt(d)")
    print("  b=2  paper  +-0.4530, +-1.5100  (over sqrt(d))")
    c2 = lloyd_max_normal(2)
    print(f"       ours   {c2[2]:+.4f}, {c2[3]:+.4f}")
    print()

    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal((a.vectors, a.d))
    x /= np.linalg.norm(x, axis=-1, keepdims=True)          # unit vectors
    q = rng.standard_normal((256, a.d))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)

    print(f"distortion on {a.vectors:,} unit vectors, d={a.d}, "
          f"against the paper's Table 1")
    print(f"  {'bits':>5}{'MSE':>12}{'paper':>10}{'ip distortion':>16}{'paper':>12}")
    paper_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    paper_ip = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}        # quoted as k/d
    for bits in (1, 2, 3, 4):
        tq = TurboQuantMSE(a.d, bits, seed=a.seed)
        codes, norm = tq.quantize(x)
        x_hat = tq.dequantize(codes, norm)
        mse = float(((x - x_hat) ** 2).sum(axis=-1).mean())
        exact_ip = q @ x.T
        approx_ip = q @ x_hat.T
        ip = float(((exact_ip - approx_ip) ** 2).mean()) * a.d
        print(f"  {bits:>5}{mse:>12.4f}{paper_mse[bits]:>10.3f}"
              f"{ip:>16.3f}{paper_ip[bits]:>12.3f}")

    print("\n'ip distortion' is E[(<q,x> - <q,x_hat>)^2] * d, which is how the")
    print("paper reports it (as a multiple of 1/d).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
