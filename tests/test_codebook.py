"""The codebook is the published Lloyd-Max optimum, and it is symmetric."""

import numpy as np

from kernel.ops.codebook import Codebook
from .harness import approx, raises

# Max & Lloyd's published optima for the unit normal. These are the numbers
# the derivation must reproduce; if the solver drifts, this catches it.
KNOWN = {
    1: [0.7980],
    2: [0.4528, 1.5104],
    3: [0.2451, 0.7560, 1.3439, 2.1520],
}


def check_matches_published_lloyd_max():
    for bits, positives in KNOWN.items():
        cb = Codebook.gaussian(bits)
        got = cb.to_float()[len(cb.to_float()) // 2:]
        approx(got, positives, tol=2e-3, what=f"Lloyd-Max b={bits}")


def check_symmetric_when_asked():
    for bits in range(1, 7):
        c = Codebook.gaussian(bits, symmetric=True).to_float()
        approx(c, -c[::-1], tol=1e-9, what=f"symmetry b={bits}")


def check_snr_gains_about_five_db_per_bit():
    """Rate-distortion sanity: a Gaussian source gains ~5 dB per bit here."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    prev = None
    for bits in range(1, 7):
        cb = Codebook.gaussian(bits)
        idx = np.searchsorted(cb.boundaries / (1 << cb.frac), x)
        snr = -10 * np.log10(((cb.to_float()[idx] - x) ** 2).mean())
        if prev is not None:
            gain = snr - prev
            assert 4.0 < gain < 6.5, f"b={bits} gained {gain:.2f} dB, expected ~5"
        prev = snr


def check_boundaries_strictly_between_centroids():
    for bits in range(1, 9):
        cb = Codebook.gaussian(bits)
        assert np.all(np.diff(cb.centroids) > 0)
        assert np.all(np.diff(cb.boundaries) > 0)
        assert np.all(cb.boundaries >= cb.centroids[:-1])
        assert np.all(cb.boundaries <= cb.centroids[1:])


def check_deterministic_across_calls():
    a = Codebook.gaussian(4, symmetric=False)
    Codebook.gaussian.cache_clear()
    b = Codebook.gaussian(4, symmetric=False)
    approx(a.centroids, b.centroids, tol=0, what="determinism")


def check_rejects_zero_bits():
    raises(ValueError, lambda: Codebook.gaussian(0), "bits=0")
