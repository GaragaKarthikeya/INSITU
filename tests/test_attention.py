"""The attention datapath: the exp LUT, the two softmaxes, and the codes claim."""

import numpy as np

from kernel.config import FixedFormat, QuantConfig
from kernel.numerics.fixed import Q
from kernel.ops.attention import (CompressedAttention, ExpLut, exact_int_matmul,
                                  reciprocal)
from kernel.ops.quantize import KVQuantizer
from .harness import approx, raises

FMT = FixedFormat()
QQ = Q(FMT.qk_width, FMT.qk_frac)


def _setup(d=64, T=64, seed=0):
    quant = QuantConfig(key_bits=3, value_bits=2)
    z = KVQuantizer(d, quant, FMT)
    a = CompressedAttention(z, FMT, quant)
    rng = np.random.default_rng(seed)
    k = z.rotation.apply(QQ.from_float(rng.standard_normal((T, d))))
    v = z.rotation.apply(QQ.from_float(rng.standard_normal((T, d))))
    q = z.rotation.apply(QQ.from_float(rng.standard_normal(d) / np.sqrt(d)))
    return z, a, z.encode(k, v), q


def check_exp_lut_accuracy():
    e = ExpLut(FMT)
    x = np.linspace(0, 12, 400)
    got = e(np.rint(x * (1 << FMT.acc_frac)).astype(np.int64)) / (1 << FMT.prob_frac)
    # An 8-bit table on the fraction gives ~2^-9 relative error.
    assert np.abs(got - np.exp(-x)).max() < 3e-3


def check_exp_lut_is_monotone_and_starts_at_one():
    e = ExpLut(FMT)
    d = np.arange(0, 1 << 20, 977)
    p = e(d)
    assert p[0] == (1 << FMT.prob_frac)
    assert np.all(np.diff(p) <= 0), "exp(-x) must be non-increasing"
    assert p[-1] == 0


def check_exp_lut_rejects_negative():
    raises(ValueError, lambda: ExpLut(FMT)(np.array([-1])), "negative delta")


def check_reciprocal():
    l = np.array([1 << FMT.prob_frac, 3 << FMT.prob_frac, 0])
    r = reciprocal(l, FMT) / (1 << FMT.recip_frac)
    approx(r[:2], [1.0, 1.0 / 3.0], tol=1e-4, what="1/l")
    assert r[2] == 0.0


def check_scoring_on_codes_equals_scoring_on_the_reconstruction():
    """The central claim: <rq, norm*centroid[code]> needs no dense key.

    Compared against explicitly rebuilding the key and dotting it, which is
    what the design refuses to do. They must agree to within the one extra
    truncation the reconstruction path performs.
    """
    z, a, kv, q = _setup()
    on_codes = QQ.to_float(a.scores(q, kv)[0])
    k_hat, _ = z.decode(kv)
    on_dense = QQ.to_float(q) @ QQ.to_float(k_hat).T
    approx(on_codes, on_dense, tol=2e-3, what="codes vs reconstruction")


def check_two_pass_matches_float_softmax():
    z, a, kv, q = _setup()
    out, _ = a.attend_two_pass(q, kv)
    k_hat, v_hat = z.decode(kv)
    s = QQ.to_float(k_hat) @ QQ.to_float(q)
    p = np.exp(s - s.max())
    ref = (p[:, None] * QQ.to_float(v_hat)).sum(0) / p.sum()
    rel = np.abs(QQ.to_float(out) - ref).max() / np.abs(ref).max()
    assert rel < 5e-3, f"relative error {rel:.2e}"


def check_online_and_two_pass_are_close_but_not_identical():
    """Both are correct; they differ by the online form's rescale truncation.

    Asserting they are EQUAL would be wrong and has to stay wrong: the online
    softmax truncates the accumulator once per rescale event. This pins the
    size of that difference so a regression in either one is visible.
    """
    z, a, kv, q = _setup(T=128, seed=5)
    o1, s1 = a.attend_two_pass(q, kv)
    o2, s2 = a.attend_online(q, kv)
    f1, f2 = QQ.to_float(o1), QQ.to_float(np.squeeze(o2))
    rel = np.abs(f1 - f2).max() / np.abs(f1).max()
    assert s2.rescale_events > 0, "test needs a case where the maximum moves"
    assert rel < 1e-2, f"online drifted {rel:.2e} from two-pass"


def check_online_reads_the_cache_once():
    """Stated as a property of the loop, not of a comment: one pass over T."""
    z, a, kv, q = _setup(T=32)
    _, stats = a.attend_online(q, kv)
    assert stats.n_tokens == 32


def check_attention_is_permutation_sensitive_in_the_online_form_only():
    """Two-pass does not care about token order; online does, via the rescales.

    This is why `KVCache.view` must return causal order, and the check exists
    so that requirement is enforced by a test rather than by a docstring.
    """
    z, a, kv, q = _setup(T=48, seed=2)
    perm = np.random.default_rng(0).permutation(48)
    shuffled = kv.select(perm)
    approx(a.attend_two_pass(q, kv)[0], a.attend_two_pass(q, shuffled)[0],
           tol=0, what="two-pass is order-free")
    o1 = np.squeeze(a.attend_online(q, kv)[0])
    o2 = np.squeeze(a.attend_online(q, shuffled)[0])
    assert not np.array_equal(o1, o2), "online form should depend on order"


def check_batched_queries_match_one_at_a_time():
    z, a, kv, q = _setup(T=40)
    rng = np.random.default_rng(9)
    qs = z.rotation.apply(QQ.from_float(rng.standard_normal((6, 64)) / 8))
    batched, _ = a.attend_two_pass(qs, kv)
    for i in range(6):
        approx(batched[i], a.attend_two_pass(qs[i], kv)[0], tol=0,
               what=f"query {i} batched vs alone")


def check_exact_int_matmul_is_exact():
    """float64 BLAS on integers is the same integer, not a close one."""
    rng = np.random.default_rng(0)
    a = rng.integers(-(1 << 20), 1 << 20, (37, 64))
    b = rng.integers(-(1 << 15), 1 << 15, (64, 53))
    approx(exact_int_matmul(a, b), a @ b, tol=0, what="exact int matmul")


def check_exact_int_matmul_refuses_to_round():
    """The one failure mode of the trick raises instead of silently rounding."""
    big = np.full((2, 4096), 1 << 30, dtype=np.int64)
    raises(OverflowError, lambda: exact_int_matmul(big, big.T), "beyond 2^53")


def check_causal_batch_is_bit_identical():
    """The batched prefill IS the token-at-a-time path, at every width.

    Not "close to" -- the same integers. If this ever weakens to a tolerance,
    the batched path has stopped being an evaluation strategy and has become a
    second implementation.
    """
    for kb, vb in ((4, 2), (3, 2), (4, 4), (6, 6), (1, 1)):
        quant = QuantConfig(key_bits=kb, value_bits=vb)
        d, T = 64, 53
        z = KVQuantizer(d, quant, FMT)
        a = CompressedAttention(z, FMT, quant)
        rng = np.random.default_rng(kb * 10 + vb)
        kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                      z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
        q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, 3, d)) / 8))
        got, _ = a.attend_causal_batch(q, kv, base=0, tile=8)
        ref = np.stack([a.attend_two_pass(q[t], kv.select(slice(0, t + 1)))[0]
                        for t in range(T)])
        approx(got, ref, tol=0, what=f"batch vs loop at {kb}/{vb} bits")


def check_batch_tile_is_invisible():
    """`tile` bounds memory and nothing else."""
    quant = QuantConfig(key_bits=4, value_bits=2)
    d, T = 64, 40
    z = KVQuantizer(d, quant, FMT)
    a = CompressedAttention(z, FMT, quant)
    rng = np.random.default_rng(0)
    kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                  z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
    q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, 2, d)) / 8))
    ref, _ = a.attend_causal_batch(q, kv, tile=T)
    for tile in (1, 3, 7, 16, 1000):
        approx(a.attend_causal_batch(q, kv, tile=tile)[0], ref, tol=0,
               what=f"tile={tile}")


def check_exp_lut_flat_matches_two_step():
    """The flattened gather is the same function, including in the tail.

    Covers the input that broke it: a delta of ~2^47, which is what masking a
    score to the format's floor and subtracting a running maximum produces.
    The unflattened form overflowed int64 there and returned a value nobody
    read; the flat form would index out of bounds. Both are now clamped, and
    the clamp is exact because everything past it maps to zero.
    """
    e = ExpLut(FMT)
    dense = np.arange(0, 1 << 22, 7, dtype=np.int64)
    extreme = np.array([e._delta_max - 1, e._delta_max, 1 << 47, 1 << 50,
                        (1 << 62)], dtype=np.int64)
    rand = np.random.default_rng(0).integers(0, 1 << 26, 200_000)
    for name, d in (("dense", dense), ("extreme", extreme), ("random", rand)):
        approx(e(d), e.two_step(d), tol=0, what=f"flat vs two-step ({name})")
    assert e(np.array([e._delta_max]))[0] == 0, "the clamp target must be zero"


def check_exp_lut_survives_a_floor_masked_score():
    """The exact input pattern the batched causal path produces."""
    e = ExpLut(FMT)
    floor = -(1 << (FMT.score_width - 1))
    delta = np.array([0, 1000, 12 - floor], dtype=np.int64)
    p = e(delta)
    assert p[0] == (1 << FMT.prob_frac) and p[2] == 0


def check_online_batch_is_bit_identical():
    """The batched online form must reproduce the per-query one exactly.

    This is the path that measures what the HARDWARE does, so "close" is not
    good enough: the whole reason the online and two-pass forms differ is the
    accumulator truncation at each rescale, and a batching that smeared those
    events would be measuring a third thing that no hardware runs.
    """
    for kb, vb in ((4, 2), (3, 2), (4, 4), (6, 6), (1, 1)):
        quant = QuantConfig(key_bits=kb, value_bits=vb)
        d, T = 64, 47
        z = KVQuantizer(d, quant, FMT)
        a = CompressedAttention(z, FMT, quant)
        rng = np.random.default_rng(kb * 31 + vb)
        kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                      z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
        q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, 3, d)) / 8))
        got, stats = a.attend_online_batch(q, kv, base=0)
        ref = np.stack([np.squeeze(a.attend_online(q[t], kv.select(slice(0, t + 1)))[0])
                        for t in range(T)])
        approx(got, ref, tol=0, what=f"online batch vs per-query at {kb}/{vb}")
        assert stats.rescale_events > 0, "test needs cases where the maximum moves"


def check_online_and_two_pass_differ_only_by_rescale_truncation():
    """They must be close, and they must NOT be identical.

    Identical would mean the rescale path is not being exercised, which would
    make every online-versus-two-pass comparison vacuous.
    """
    quant = QuantConfig(key_bits=4, value_bits=2)
    d, T = 64, 96
    z = KVQuantizer(d, quant, FMT)
    a = CompressedAttention(z, FMT, quant)
    rng = np.random.default_rng(11)
    kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                  z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
    q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, 2, d)) / 8))
    on, st = a.attend_online_batch(q, kv)
    tp, _ = a.attend_causal_batch(q, kv, tile=16)
    assert st.rescale_events > 0
    assert not np.array_equal(on, tp), "rescale path not exercised"
    rel = np.abs(QQ.to_float(on) - QQ.to_float(tp)).max() / \
        max(np.abs(QQ.to_float(tp)).max(), 1e-9)
    assert rel < 0.05, f"online drifted {rel:.2e} from two-pass"
