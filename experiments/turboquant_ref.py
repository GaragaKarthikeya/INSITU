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


class TurboQuantProd:
    """Algorithm 2 of the paper: MSE at b-1 bits, plus a 1-bit QJL correction
    on the residual, for an unbiased inner-product estimator.

    Why this is unbiased and TurboQuantMSE is not
    -----------------------------------------------
    For a fixed unit vector `u` and a random Gaussian projection `s ~ N(0, I)`,
    let `g = s . u` (unit variance) and `h = s . y` for some other vector `y`.
    `g` and `h` are jointly Gaussian with correlation `rho = <u, y> / ||y||`, and
    for jointly Gaussian variables `E[sign(g) h] = sqrt(2/pi) * rho`. So

        E[sign(s . u) * (s . y)] = sqrt(2/pi) * <u, y>

    Averaging over `d` independent rows of `S` (i.e. `S` is d x d, one sign bit
    per row) gives a single-sample Monte Carlo estimate of `<u, y>` that is
    unbiased in expectation over `S`, with variance shrinking as d grows. Apply
    that to `u = r / ||r||`, the *residual* direction left over after the MSE
    step, and the correction terms add: `Quant_mse` alone is missing exactly
    the part of `<x, y>` that lives in the residual, and this recovers it
    without bias -- at the cost of `||r||` (one more norm) and `d` sign bits.

    Only the keys are meant to use this (paper: TurboQuant_prod for keys,
    TurboQuant_mse for values, since values are never dotted with anything).
    """

    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 2:
            raise ValueError(f"bits={bits} must be >= 2 (1 for the mse part, 1 for qjl)")
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d, bits - 1, seed=seed)
        # A fresh random matrix, independent of the rotation `self.mse.R` uses.
        self.S = np.random.default_rng(seed + 97).standard_normal((d, d))

    def quantize(self, x):
        """(..., d) real -> (idx, qjl, norm, r_norm)."""
        x = np.asarray(x, dtype=np.float64)
        idx, norm = self.mse.quantize(x)
        x_hat_mse = self.mse.dequantize(idx, norm)
        r = x - x_hat_mse
        r_norm = np.linalg.norm(r, axis=-1)
        u_r = np.divide(r, r_norm[..., None], out=np.zeros_like(r), where=r_norm[..., None] > 0)
        qjl = np.sign(u_r @ self.S.T)
        qjl[qjl == 0] = 1.0          # sign(0) picked arbitrarily; never a tie in practice
        return idx, qjl, norm, r_norm

    def dequantize(self, idx, qjl, norm, r_norm):
        """Codes, qjl signs and both norms -> the reconstruction, original basis."""
        x_hat_mse = self.mse.dequantize(idx, norm)
        dir_est = (np.sqrt(np.pi / 2) / self.d) * (qjl @ self.S)
        return x_hat_mse + r_norm[..., None] * dir_est

    def score(self, q, idx, qjl, norm, r_norm):
        """The paper's inner product: reconstruct first, then dot."""
        return np.asarray(q, dtype=np.float64) @ self.dequantize(idx, qjl, norm, r_norm).T


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

    print(f"\nkeys: does the 1-bit QJL correction (TurboQuant_prod) beat plain")
    print(f"TurboQuant_mse on inner-product distortion, at the SAME total bits/dim?")
    print(f"(mse@b vs prod@b = mse@(b-1) + 1 qjl bit; prod pays one extra float norm)")
    print(f"  {'bits':>5}{'mse ip':>12}{'prod ip':>12}{'ratio':>10}{'mse MSE':>12}{'prod MSE':>12}")
    for bits in (2, 3, 4, 5, 6):
        mse = TurboQuantMSE(a.d, bits, seed=a.seed)
        codes, norm = mse.quantize(x)
        x_hat_mse = mse.dequantize(codes, norm)
        mse_ip = float(((q @ x.T - q @ x_hat_mse.T) ** 2).mean()) * a.d
        mse_mse = float(((x - x_hat_mse) ** 2).sum(axis=-1).mean())

        prod = TurboQuantProd(a.d, bits, seed=a.seed)
        idx, qjl, pnorm, r_norm = prod.quantize(x)
        x_hat_prod = prod.dequantize(idx, qjl, pnorm, r_norm)
        prod_ip = float(((q @ x.T - q @ x_hat_prod.T) ** 2).mean()) * a.d
        prod_mse = float(((x - x_hat_prod) ** 2).sum(axis=-1).mean())

        print(f"  {bits:>5}{mse_ip:>12.4f}{prod_ip:>12.4f}{mse_ip / prod_ip:>10.2f}x"
              f"{mse_mse:>12.4f}{prod_mse:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
