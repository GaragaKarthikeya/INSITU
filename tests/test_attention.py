"""The attention datapath: the exp LUT, the two softmaxes, and the codes claim."""

import numpy as np

from kernel.config import FixedFormat, QuantConfig
from kernel.numerics.fixed import Q
from kernel.ops.attention import (CompressedAttention, ExpLut, exact_int_matmul,
                                  reciprocal)
from kernel.ops.quantize import KVQuantizer
from .harness import approx, exact, raises

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
    r, sh = reciprocal(l, FMT)
    # `out = (acc*recip) >> shift` realises `acc * 2**prob_frac / l`, so
    # `recip / 2**shift` is `2**prob_frac / l` -- which is 1.0 when l is unity
    # in Q(prob_frac). The old fixed-Q form divided by `2**recip_frac` here.
    approx(r[:2] / (2.0 ** sh[:2]), [1.0, 1.0 / 3.0], tol=1e-4, what="1/l")
    assert r[2] == 0


def check_the_reciprocal_holds_its_precision_at_every_context():
    """The check that was missing, and the reason limit 9 went unnoticed.

    The old fixed-Q16 divide was `2**16 / T` levels: 2 at ctx 32,768 and 0 at
    131,072. Its error was the fractional part of `2**31/l`, which made it a
    lottery: 0.2% at powers of two and 25% at ctx 24,576. Every context this
    project tested was a power of two, so every measurement landed on a winning
    ticket and nothing failed.

    So this sweeps the unlucky contexts on purpose. `l = T * 2**15` is the
    measured law: near-uniform attention makes every `p` near unity.
    """
    for T in (64, 1024, 4096, 16384, 20000, 24576, 32768, 49152, 65536,
              131072, 262144, 1 << 22):
        l = int(round(T * (1 << FMT.prob_frac) * 0.998))
        r, sh = reciprocal(np.array([l]), FMT)
        r, sh = int(r[0]), int(sh[0])
        got = r / 2.0 ** sh
        want = (1 << FMT.prob_frac) / l
        assert abs(got - want) / want < 1e-6, (T, got, want)
        # The quotient is normalised, so it is 31 bits whatever the context --
        # that is the property, not the error bound that follows from it.
        assert 1 << 30 < r <= 1 << 31, (T, r)
        # And `acc * recip` must stay inside 64 bits: acc is ACC_WIDTH signed.
        assert r * (1 << (FMT.acc_width - 1)) < (1 << 63), (T, r)


def check_the_reciprocal_shift_is_never_a_left_shift():
    """`rshift` refuses a negative amount, and small `l` is the case that tries.

    A masked position gives `l = 0` and a one-token context gives the smallest
    real `l` there is; both must still produce a non-negative shift.
    """
    l = np.array([0, 1, 1 << FMT.prob_frac, (1 << 47) - 1])
    _, sh = reciprocal(l, FMT)
    assert np.all(sh >= 0), sh


def check_the_reciprocal_width_is_one_number_everywhere():
    """The divisor width is defined in three places and they must all agree.

    `FixedFormat.recip_norm_bits`, `RECIP_NORM_BITS` in `ops/attention.py` and
    `RECIP_BITS` in `rtl/finalize.sv` all describe the same thing: the window
    `l` is normalised into before the divide. The RTL derives it as
    `RECIP_FRAC + PROB_FRAC + 1`, which happens to be 32 as well.

    Happening to be right is the problem. Those two fields no longer take any
    part in the arithmetic, so someone changing `prob_frac` -- an ordinary
    thing to want to do -- would resize the hardware's divider while leaving
    the model's alone, and the two would divide differently with nothing
    saying so. Bit-exactness would not catch it either: the goldens come from
    this side.
    """
    import pathlib
    import re

    from kernel.ops.attention import RECIP_NORM_BITS

    exact(FMT.recip_norm_bits, RECIP_NORM_BITS,
          "FixedFormat and ops/attention disagree on the reciprocal width")

    rtl = (pathlib.Path(__file__).resolve().parents[1] / "rtl" / "finalize.sv").read_text()
    frac = int(re.search(r"parameter int RECIP_FRAC\s*=\s*(\d+)", rtl).group(1))
    prob = int(re.search(r"parameter int PROB_FRAC\s*=\s*(\d+)", rtl).group(1))
    exact(frac + prob + 1, RECIP_NORM_BITS,
          "rtl/finalize.sv sizes its divider differently from the model")


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

    Asserting they are equal would be wrong, and has to stay wrong: the online
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
    """The batched prefill is the token-at-a-time path, at every width.

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


def check_causal_batch_dense_is_bit_identical():
    """`attend_causal_batch_dense` is architecture A's token loop, not close to it.

    Same guarantee as `check_causal_batch_is_bit_identical`, for the other
    architecture: the loop calls `attend_two_pass` once per query against a
    growing prefix (`ctx = t+1`), re-decoding the cache from scratch every
    time; the batched path decodes once and reuses it. If those ever drift
    apart it means the per-term truncation order stopped matching, not that
    the batched path found a faster but different answer.
    """
    for kb, vb in ((4, 2), (3, 2), (4, 4), (6, 6), (1, 1)):
        quant = QuantConfig(key_bits=kb, value_bits=vb)
        d, T = 64, 53
        z = KVQuantizer(d, quant, FMT)
        a = CompressedAttention(z, FMT, quant)
        a.score_mode = "dense"
        rng = np.random.default_rng(kb * 10 + vb)
        kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                      z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
        q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, 3, d)) / 8))
        q_unrot = QQ.from_float(rng.standard_normal((T, 3, d)) / 8)

        a._q_unrot = q_unrot
        got, _ = a.attend_causal_batch_dense(q, kv, base=0, tile=8, kchunk=16)

        ref = np.empty_like(got)
        for t in range(T):
            a._q_unrot = q_unrot[t]
            ref[t], _ = a.attend_two_pass(q[t], kv.select(slice(0, t + 1)))
        approx(got, ref, tol=0, what=f"dense batch vs loop at {kb}/{vb} bits")


def check_causal_batch_dense_rejects_fused():
    quant = QuantConfig(key_bits=4, value_bits=2)
    d, T = 64, 8
    z = KVQuantizer(d, quant, FMT)
    a = CompressedAttention(z, FMT, quant)
    rng = np.random.default_rng(0)
    kv = z.encode(z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))),
                  z.rotation.apply(QQ.from_float(rng.standard_normal((T, d)))))
    q = z.rotation.apply(QQ.from_float(rng.standard_normal((T, d))))
    raises(ValueError, lambda: a.attend_causal_batch_dense(q, kv))


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

    This is the path that measures what the hardware does, so "close" is not
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
    """They must be close, and they must not be identical.

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
