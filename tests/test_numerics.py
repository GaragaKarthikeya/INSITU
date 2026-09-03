"""Fixed-point and floating-point primitives."""

import numpy as np

from kernel.numerics import fp
from kernel.numerics.fixed import Q, fwht, inv_sqrt_q15, lshift, rshift, sqrt_shift
from .harness import approx, exact, raises


def _dense_hadamard(n):
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def check_fwht_matches_dense_hadamard():
    """The butterfly is the matrix, exactly, at every power-of-two size."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 4, 8, 16, 64, 256):
        v = rng.integers(-1000, 1000, n)
        exact(fwht(v), _dense_hadamard(n) @ v, f"fwht n={n}")


def check_fwht_rejects_non_power_of_two():
    raises(ValueError, lambda: fwht(np.arange(6)), "fwht(6)")


def check_rounding_modes_differ_where_they_should():
    """floor, half-to-even and half-up agree away from ties and differ on them."""
    x = np.arange(-8, 9)
    exact(rshift(x, 1, "floor"), x >> 1, "floor is the operator")
    # Ties: odd values shifted by 1. even rounds to the even neighbour.
    exact(rshift(np.array([1, 3, 5, 7]), 1, "even"), np.array([0, 2, 2, 4]), "half-to-even")
    exact(rshift(np.array([1, 3, 5, 7]), 1, "up"), np.array([1, 2, 3, 4]), "half-up")
    # Half-to-even is unbiased over a symmetric tie set; floor is not.
    ties = np.arange(-101, 102, 2)
    assert abs(int(rshift(ties, 1, "even").sum())) <= 1
    assert int((ties >> 1).sum()) < -50


def check_rshift_zero_is_identity_not_error():
    exact(rshift(np.array([-3, 7]), 0, "even"), np.array([-3, 7]), "rshift 0")


def check_q_roundtrip_and_saturation():
    q = Q(16, 8)
    x = np.array([0.0, 1.0, -1.0, 0.00390625, 127.996])
    approx(q.to_float(q.from_float(x)), x, tol=1.0 / 256, what="Q8.8 round trip")
    assert q.from_float(np.array([1e6]))[0] == q.hi
    assert q.from_float(np.array([-1e6]))[0] == q.lo
    assert q.overflow_count(np.array([q.hi + 1, 0, q.lo - 1])) == 2


def check_q_wraps_when_not_saturating():
    q = Q(8, 0, saturate=False)
    exact(q.clamp(np.array([128, -129, 0])), np.array([-128, 127, 0]), "wrap")


def check_fixed_ops_reject_floats():
    raises(TypeError, lambda: rshift(np.array([1.5]), 1), "rshift on float")


def check_sqrt_shift_only_for_even_powers():
    assert sqrt_shift(1) == 0 and sqrt_shift(4) == 1 and sqrt_shift(64) == 3
    assert sqrt_shift(2) is None and sqrt_shift(32) is None
    assert sqrt_shift(6) is None
    assert inv_sqrt_q15(32) == 5793      # round(2^15 / sqrt(32))


def check_fp16_accumulates_in_fp32():
    """A long reduction must not lose the small terms the way pure fp16 would."""
    n = 4096
    w = np.full((1, n), 1.0, dtype=np.float16)
    x = np.full(n, 1.0 / 1024, dtype=np.float16)
    got = float(fp.matvec(w, x)[0])
    approx(got, n / 1024, tol=1e-3, what="fp32 accumulation")
    # The same sum performed in fp16 saturates its mantissa and stalls.
    naive = np.float16(0)
    for v in x:
        naive = np.float16(naive + v)
    assert float(naive) < got - 0.5, "fp16 accumulation should have stalled"


def check_ordered_matvec_matches_when_chunk_is_whole():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 512)).astype(np.float32)
    x = rng.standard_normal(512).astype(np.float32)
    exact(fp.matvec_ordered(w, x, 512), fp.matvec(w, x), "chunk == k")


def check_fp16_range_check_fires():
    raises(ValueError, lambda: fp.check_fp16_range(np.array([1e5]), "W"), "overflow")
    raises(ValueError, lambda: fp.check_fp16_range(np.array([np.inf]), "W"), "inf")
