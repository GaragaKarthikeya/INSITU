"""The rotation's three load-bearing properties, each checked directly."""

import numpy as np

from kernel.numerics.fixed import Q
from kernel.ops.rotate import Rotation
from .harness import approx, exact, raises


def check_orthonormal():
    for d in (8, 32, 64, 128, 256):
        m = Rotation.from_seed(d, 3).matrix()
        approx(m @ m.T, np.eye(d), tol=1e-12, what=f"R R^T = I, d={d}")


def check_preserves_inner_products():
    """The property the whole architecture rests on: <Rq, Rk> == <q, k>.

    Checked in fixed point, not just in real arithmetic, because that is where
    it could fail: the 1/sqrt(d) scale is the one truncation in the rotation.
    """
    d, q = 64, Q(32, 16)
    r = Rotation.from_seed(d, 0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.standard_normal(d), rng.standard_normal(d)
        ra, rb = r.apply(q.from_float(a)), r.apply(q.from_float(b))
        approx(q.to_float(ra) @ q.to_float(rb), a @ b, tol=2e-3, what="<Rq,Rk>")


def check_fixed_matches_dense_matrix():
    d, q = 64, Q(32, 16)
    r = Rotation.from_seed(d, 7)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, d))
    approx(q.to_float(r.apply(q.from_float(x))), x @ r.matrix().T,
           tol=1e-3, what="fixed vs dense")


def check_scale_is_an_exact_shift_only_for_even_powers():
    assert Rotation.from_seed(64, 0).shift == 3
    assert Rotation.from_seed(256, 0).shift == 4
    assert Rotation.from_seed(32, 0).shift is None      # odd power -> Q15 multiply
    assert Rotation.from_seed(128, 0).shift is None


def check_seed_changes_the_transform():
    d = 64
    a, b = Rotation.from_seed(d, 0), Rotation.from_seed(d, 1)
    assert not np.array_equal(a.signs, b.signs)
    assert a.signs.shape[1] == d
    two = Rotation.from_seed(d, 0, rounds=2)
    assert not np.array_equal(two.signs[0], two.signs[1]), "rounds must be independent"
    assert np.abs(a.matrix() - b.matrix()).max() > 0.1


def check_seed_is_reproducible():
    exact(Rotation.from_seed(64, 42).signs, Rotation.from_seed(64, 42).signs, "PCG64")


def check_rejects_non_power_of_two():
    raises(ValueError, lambda: Rotation.from_seed(48, 0), "d=48")


def check_default_is_one_round():
    """The default is a measured choice; changing it should require changing this."""
    from kernel.config import QuantConfig
    assert QuantConfig().rot_rounds == 1
    assert Rotation.for_quant(QuantConfig(), 64).rounds == 1


def check_gaussianises_outlier_heavy_activations():
    """The claim `Codebook.gaussian` depends on, measured rather than assumed.

    LLM activations are outlier-heavy: a few channels carry far more energy
    than the rest, which is the worst case for a scalar quantizer. Two rounds
    must bring the excess kurtosis close to the Gaussian's zero.
    """
    d = 128
    r = Rotation.from_seed(d, 0, rounds=2)   # the two-round transform, on purpose
    rng = np.random.default_rng(0)
    before, after = [], []
    for _ in range(300):
        x = rng.standard_normal(d)
        x[rng.choice(d, 4, replace=False)] *= 30.0
        before.append(_kurtosis(x))
        after.append(_kurtosis(r.matrix() @ x))
    b, a = float(np.mean(before)), float(np.mean(after))
    assert b > 20, f"outlier-heavy input should be leptokurtic, got {b:.1f}"
    assert abs(a) < 0.4, f"rotated should be near-Gaussian, excess kurtosis {a:.2f}"


def check_one_round_overshoots_into_platykurtic():
    """One round is measurably less Gaussian, and still the better default.

    Kept because the kurtosis fact is real and worth knowing; the point is that
    it does not predict quality. Measured end to end, one round beats two by
    1.3% perplexity. This test pins the distributional claim so nobody
    "rediscovers" it and flips the default back on that basis alone.

    `H D` on a one-hot vector gives `+-1/sqrt(d)` in every channel: a two-point
    distribution, excess kurtosis exactly -2. One round spreads the energy but
    does not randomise the magnitudes, and a platykurtic source is as badly
    matched to a Gaussian codebook as a leptokurtic one.
    """
    d = 128
    rng = np.random.default_rng(0)
    one_hot = np.eye(d)[rng.integers(d, size=200)]

    k1 = np.mean([_kurtosis(v) for v in one_hot @ Rotation.from_seed(d, 0, 1).matrix().T])
    k2 = np.mean([_kurtosis(v) for v in one_hot @ Rotation.from_seed(d, 0, 2).matrix().T])
    assert k1 < -1.5, f"one round should overshoot to about -2, got {k1:.2f}"
    assert abs(k2) < abs(k1) / 4, f"two rounds should recover: {k1:.2f} -> {k2:.2f}"


def check_extra_rounds_stay_orthonormal():
    for rounds in (1, 2, 3):
        m = Rotation.from_seed(64, 1, rounds).matrix()
        approx(m @ m.T, np.eye(64), tol=1e-12, what=f"rounds={rounds}")


def check_rejects_zero_rounds():
    raises(ValueError, lambda: Rotation.from_seed(64, 0, 0), "rounds=0")


def _kurtosis(x):
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2 - 3.0)
