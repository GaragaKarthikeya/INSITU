"""The golden vectors, and the properties the RTL will be trusted against.

Every check here is really one question: **can a testbench that passes against
these files be wrong?** The packing round trip, the beat count, the group
order and the determinism are the four ways it could be, so they are the four
things asserted.
"""

import functools
import json
import os
import tempfile

import numpy as np

from kernel import FixedFormat, ModelConfig, QuantConfig
from kernel.hw.vectors import (BEAT_BITS, beats, build, collect, emit,
                               pack_lanes, unpack_lanes)
from .harness import exact, raises


@functools.lru_cache(maxsize=None)
def _built(ctx=32, seed=0):
    return build(ctx=ctx, seed=seed)


def check_24bit_packing_round_trips():
    """The exit criterion of step 3.

    `attn_ingress.sv` is a shift register because 24 does not divide 512, so a
    lane straddles a beat boundary. That is exactly the arithmetic that is easy
    to get one bit wrong in and hard to notice: an off-by-one shift produces a
    stream of the right length, full of plausible-looking values. Round-tripping
    the full signed range, at every alignment, is what rules it out.
    """
    q = FixedFormat()
    bits = q.qk_width
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    # Every value that can appear on a lane, at the extremes and at random.
    rng = np.random.default_rng(0)
    vals = np.concatenate([
        np.array([lo, lo + 1, -1, 0, 1, hi - 1, hi], dtype=np.int64),
        rng.integers(lo, hi + 1, size=1024, dtype=np.int64),
    ])
    exact(unpack_lanes(pack_lanes(vals, bits), bits, vals.size), vals,
          "24-bit lanes")

    # And at every start alignment, because a lane's bit offset within a byte
    # cycles with period 8 / gcd(24, 8) -- a packer can be right for one
    # alignment and wrong for the next.
    for n in range(1, 25):
        v = rng.integers(lo, hi + 1, size=n, dtype=np.int64)
        exact(unpack_lanes(pack_lanes(v, bits), bits, n), v, f"n={n}")

    # 64 values x 24 b = 1,536 b = exactly 3 beats of 512. No TKEEP, no DRE.
    one_vector = pack_lanes(rng.integers(lo, hi + 1, size=64, dtype=np.int64), bits)
    assert len(one_vector) == 192
    assert len(beats(one_vector)) == 3
    assert all(len(b) == BEAT_BITS // 4 for b in beats(one_vector))


def check_other_lane_widths_round_trip_too():
    """The cache row uses 4-bit and 2-bit lanes through the same packer.

    A packer that only works at a byte-aligned width is a packer that has not
    been tested; `KEY_BITS` and `VAL_BITS` are the widths that actually exercise
    the sub-byte path.
    """
    rng = np.random.default_rng(1)
    for bits in (2, 3, 4, 9, 16, 17, 24):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        v = rng.integers(lo, hi + 1, size=97, dtype=np.int64)
        exact(unpack_lanes(pack_lanes(v, bits), bits, v.size), v, f"{bits} b")
    raises(ValueError, lambda: pack_lanes([1], 0), "zero-width lanes")


def check_beat_hex_is_msb_first_over_a_little_endian_stream():
    """Byte 0 of the stream is bits [7:0] of beat 0, so a beat prints reversed.

    The single most common way a testbench and a host packer disagree while
    each looks correct on its own.
    """
    buf = bytes(range(64))
    line = beats(buf)[0]
    assert line.startswith("3f3e3d"), line
    assert line.endswith("020100"), line
    # A short tail is zero-padded up to a whole beat, not silently dropped.
    assert len(beats(bytes([0xAB]))) == 1
    assert beats(bytes([0xAB]))[0].endswith("ab")


def check_the_stream_is_eight_groups_of_six_with_k_and_v_first():
    """Group layout and, more importantly, the order INSIDE a group.

    The current token attends to itself, so its key and value must reach the
    cache before any score is computed. `plan.MD` calls this causality rather
    than preference, and an inverted order produces an off-by-one-token
    attention that still runs.
    """
    vs, _ = _built()
    groups = vs.ingress_groups()
    assert len(groups) == vs.model.num_kv_heads == 8
    assert all(g.shape == (2 + vs.model.kv_groups, 64) for g in groups)

    for h, g in enumerate(groups):
        exact(g[0], vs.wire["k"][0, h], f"group {h} slot 0 must be k")
        exact(g[1], vs.wire["v"][0, h], f"group {h} slot 1 must be v")
        for j in range(vs.model.kv_groups):
            exact(g[2 + j], vs.wire["q"][0, h * vs.model.kv_groups + j],
                  f"group {h} slot {2 + j}")

    # 9,216 B down and 1,152 B per group -- the numbers the plan is sized from.
    assert len(vs.ingress_bytes()) == 9216
    assert len(pack_lanes(groups[0], vs.lane_bits)) == 1152
    assert len(vs.ingress_beats()) == 48 * vs.beats_per_vector == 144


def check_the_stream_decodes_back_to_the_tapped_vectors():
    """The whole point: what the RTL will receive IS what the model computed."""
    vs, _ = _built()
    d = vs.model.head_dim
    flat = unpack_lanes(vs.ingress_bytes(), vs.lane_bits, 48 * d).reshape(48, d)
    exact(np.concatenate(vs.ingress_groups(), axis=0), flat, "ingress round trip")


def check_the_cache_row_is_the_packed_buffer_byte_for_byte():
    """`kv_store_ddr.sv` must match `ops/quantize.py`'s layout exactly.

    The row is taken from the cache's own buffer rather than re-packed here,
    so the vectors cannot describe a format the model does not write.
    """
    vs, k = _built()
    assert vs.rows.shape == (8, 52), vs.rows.shape       # 32 + 16 + 2 + 2 B
    exact(vs.rows, k.cache.buf[vs.context], "row is the cache's own bytes")
    # And it decodes through the wire format, not just the shadow.
    kv = k.cache.quantizer.unpack(vs.rows)
    exact(kv.k_idx, k.cache._k_idx[vs.context], "k codes")
    exact(kv.v_norm, k.cache._v_norm[vs.context], "v norm")


def check_the_softmax_golden_is_online_and_differs_from_two_pass():
    """`plan.MD`: the TB golden is `attend_online`, never `attend_two_pass`.

    They are not bit-identical -- the online form truncates the accumulator at
    every rescale -- so a testbench pointed at the wrong file can never pass.
    Both are emitted; the manifest names which one is golden.
    """
    vs, _ = _built()
    assert vs.out_online is not None and vs.out_two_pass is not None
    assert vs.out_online.shape == vs.out_two_pass.shape
    # Close, because it is the same arithmetic in a different order...
    assert np.abs(vs.out_online - vs.out_two_pass).max() < (1 << 12)
    # ...and not equal, which is the reason the distinction has to be made.
    assert not np.array_equal(vs.out_online, vs.out_two_pass), \
        "if these ever agree, the online path stopped rescaling"


def check_scores_cover_the_whole_causal_context():
    """One score per (query head, cached token), the new token included."""
    vs, _ = _built()
    assert vs.scores.shape == (vs.model.num_heads, vs.context + 1)


def check_vectors_regenerate_deterministically_from_a_seed():
    """The other half of step 3's exit criterion.

    Two builds from the same seed must be byte-identical, and a different seed
    must actually move -- a generator that ignored its seed would pass the
    first half alone.
    """
    a, _ = build(ctx=8, seed=3)
    b, _ = build(ctx=8, seed=3)
    c, _ = build(ctx=8, seed=4)
    assert a.ingress_bytes() == b.ingress_bytes()
    exact(a.out_online, b.out_online, "same seed, same answer")
    exact(a.rows, b.rows, "same seed, same cache row")
    assert a.ingress_bytes() != c.ingress_bytes(), "the seed must do something"


def check_emit_writes_a_complete_and_reloadable_set():
    vs, _ = _built()
    with tempfile.TemporaryDirectory() as d:
        paths = emit(vs, d)
        names = {os.path.basename(p) for p in paths}
        assert {"ingress.hex", "kv_rows.hex", "scores.hex", "out.hex",
                "out.online.hex", "rot_signs.hex", "rot_random_in.hex",
                "rot_random_out.hex", "manifest.json"} <= names
        assert all(f"ingress_group{i}.hex" in names for i in range(8))

        man = json.load(open(os.path.join(d, "manifest.json")))
        assert man["softmax_golden"] == "out.online.hex"
        assert man["beats_per_vector"] == 3 and man["lane_bits"] == 24
        assert man["row_bytes"] == 52 and man["context"] == vs.context
        assert man["rot_sign_negative_is_one"] is True
        assert man["rot_random_vectors"] >= 1000

        lines = open(os.path.join(d, "ingress.hex")).read().split()
        assert len(lines) == 144
        # The file reloads to the same bytes the stream had -- a hex writer
        # that dropped a nibble would still look like a plausible file.
        got = b"".join(bytes(reversed(bytes.fromhex(ln))) for ln in lines)
        assert got == vs.ingress_bytes()


def check_the_tap_is_an_observer_and_leaves_no_trace():
    """A debug callback that changed a value would poison every vector below it.

    So: the same step run with and without the observer must return bit-identical
    output, and the kernel must be left as it was found.
    """
    from kernel import AttentionKernel, KernelConfig

    def one(attach):
        m = ModelConfig(hidden_size=256, num_heads=8, num_kv_heads=2, head_dim=32)
        k = AttentionKernel(KernelConfig(model=m, quant=QuantConfig()), capacity=8)
        x = np.random.default_rng(7).standard_normal((3, 256)).astype(np.float32) * 0.05
        if attach:
            vs = collect(k, x)
            assert k.on_stage is None, "the observer must be detached again"
            return vs.out_two_pass
        y, _ = k.forward(x)
        k2 = AttentionKernel(KernelConfig(model=m, quant=QuantConfig()), capacity=8)
        seen = {}
        k2.on_stage = lambda n, v: seen.setdefault(n, v)
        k2.forward(x)
        return seen["out"]

    exact(one(True), one(False), "observer changed the answer")


# --------------------------------------------------------------------------
# step 6: the encoder stimulus
# --------------------------------------------------------------------------

def check_encoder_stimulus_contains_exact_boundary_ties():
    """The reason the tie vectors are constructed instead of sampled.

    `_encode_plane` compares with `>`, so a channel exactly on a boundary falls
    to the LOWER bin. An RTL comparator that used `>=` would disagree on those
    channels alone -- and a tie needs `x << 8 == norm * boundary` exactly, which
    random stimulus hits with probability about 2**-32. If this check ever finds
    no ties, `tb_rot_encode` has silently stopped testing the rule.
    """
    from kernel.hw.vectors import ENC_TIES, encoder_stimulus

    vs, _ = _built()
    qz, plane = vs.quantizer, vs.quantizer.key
    x = encoder_stimulus(vs)
    norms = qz._norm_wire(x)
    shift = plane.threshold_frac - plane.qk_frac
    thresholds = norms[:, None] * plane.codebook.boundaries[None, :]

    ties = ((x[:, -1, None] << shift) == thresholds).any(axis=1)
    assert ties.sum() >= 8, f"only {int(ties.sum())} exact ties in {x.shape[0]} vectors"
    assert ties[-ENC_TIES:].sum() >= 8, "the constructed tail is not where the ties are"

    # And the rule is load-bearing: `>=` would move those channels by one bin.
    t3 = thresholds[:, None, :]
    lo = ((x[..., None] << shift) > t3).sum(axis=-1)
    hi = ((x[..., None] << shift) >= t3).sum(axis=-1)
    assert np.any(lo != hi), "no channel distinguishes `>` from `>=`"


def check_encoder_goldens_come_from_the_kernels_own_quantizer():
    """Not a second codebook built here from the same config.

    A rebuilt one agrees today and drifts the first time `Codebook.build`
    changes, which would make the RTL bit-exact against a stale format.
    """
    vs, kernel = _built()
    assert vs.quantizer is kernel.cache.quantizer


def check_encoder_files_reload_to_the_quantizers_own_answer():
    from kernel.hw.vectors import BOUND_BITS, encoder_stimulus

    vs, _ = _built()
    qz = vs.quantizer
    with tempfile.TemporaryDirectory() as d:
        names = {os.path.basename(p) for p in emit(vs, d)}
        assert {"enc_in.hex", "enc_norm.hex", "enc_idx_key.hex",
                "enc_idx_value.hex", "enc_bounds_key.hex",
                "enc_bounds_value.hex"} <= names

        man = json.load(open(os.path.join(d, "manifest.json")))
        x = encoder_stimulus(vs)
        assert man["enc_vectors"] == x.shape[0] == 16 + man["enc_random"] + 32

        def rows(name, width):
            out = []
            for ln in open(os.path.join(d, name)).read().split():
                v = [int(ln[i:i + width], 16) for i in range(0, len(ln), width)]
                out.append(list(reversed(v)))     # lane 0 is printed rightmost
            return np.array(out, dtype=np.int64)

        sign = 1 << (vs.lane_bits - 1)
        got_x = (rows("enc_in.hex", 6) ^ sign) - sign
        exact(got_x, x, "enc_in.hex round trip")

        norms = qz._norm_wire(x)
        exact(rows("enc_norm.hex", 4).ravel(), norms, "enc_norm.hex")
        for plane in (qz.key, qz.value):
            exact(rows(f"enc_idx_{plane.name}.hex", 2),
                  qz._encode_plane(x, plane, norms), f"enc_idx_{plane.name}.hex")

            bsign = 1 << (BOUND_BITS - 1)
            b = (rows(f"enc_bounds_{plane.name}.hex", BOUND_BITS // 4).ravel()
                 ^ bsign) - bsign
            exact(b, plane.codebook.boundaries, f"enc_bounds_{plane.name}.hex")


def check_the_encoder_goldens_agree_with_the_packed_cache_row():
    """The 8 real key vectors are the first 8 lines of `enc_idx_key.hex`.

    `kv_rows.hex` is what the cache stored and `enc_idx_key.hex` is what the
    encoder bench checks against. If those two disagree, one of the benches is
    verifying a row layout the other will never produce.
    """
    vs, _ = _built()
    qz = vs.quantizer
    x = np.concatenate([vs.rotated["k"][0], vs.rotated["v"][0]], axis=0)
    norms = qz._norm_wire(x)
    kv = qz.unpack(vs.rows)
    exact(kv.k_idx, qz._encode_plane(x[:8], qz.key, norms[:8]), "row k codes")
    exact(kv.v_idx, qz._encode_plane(x[8:], qz.value, norms[8:]), "row v codes")
    exact(kv.k_norm, norms[:8], "row k norm")
    exact(kv.v_norm, norms[8:], "row v norm")


def check_the_cache_rows_are_the_rows_the_scores_were_computed_over():
    """`cache_rows_hN.hex` and `scores.hex` must describe one cache, not two.

    The score bench streams the rows and compares against the scores. If the
    rows were snapshotted at a different point in the step than the scores
    were computed at -- before `append_rotated` rather than after, say -- the
    bench would be off by one token and would still look like a datapath bug.
    """
    vs, kernel = _built()
    assert vs.cache.shape[0] == vs.context + 1
    for h in range(vs.model.num_kv_heads):
        exact(vs.cache[:, h], kernel.cache.buf[:vs.context + 1, h],
              f"cache rows head {h}")
        kv = kernel.cache.quantizer.unpack(vs.cache[:, h])
        view = kernel.cache.view(h)
        exact(kv.k_idx, view.k_idx, f"head {h} k codes")
        exact(kv.k_norm, view.k_norm, f"head {h} k norms")
    # The last row of the cache is the token `kv_rows.hex` holds, so the two
    # score-lane stimuli cannot describe different tokens.
    exact(vs.cache[vs.context], vs.rows, "the last cache row is kv_rows.hex")


def check_the_product_table_is_the_multiply_the_scan_avoids():
    """`qtab_table.hex` must reproduce `scores` when gathered by real codes.

    The table is only worth building if summing its entries at a token's codes
    is the same integer as the dot product `CompressedAttention.scores`
    computes. Checked here on real cache rows rather than asserted, because
    the whole architecture rests on that identity.
    """
    from kernel.hw.vectors import CENT_BITS

    vs, kernel = _built()
    cents = kernel.cache.quantizer.key.codebook.centroids
    prod_w = vs.lane_bits + CENT_BITS - 1
    assert np.abs(np.outer(vs.rotated["q"][0], cents)).max() < (1 << (prod_w - 1))

    groups = vs.model.kv_groups
    for head in (0, 7, 31):
        q = vs.rotated["q"][0][head]
        table = q[:, None] * cents[None, :]              # (d, 2**key_bits)
        view = kernel.cache.view(head // groups)
        gathered = table[np.arange(vs.model.head_dim), view.k_idx.astype(np.int64)]
        s, _ = kernel.attn.scores(q, view)
        from kernel.numerics.fixed import rshift
        exact(rshift(gathered.sum(axis=-1) * view.k_norm,
                     kernel.attn.score_shift), s, f"head {head} gathered scores")


def check_the_constructed_norm_rows_are_negative_where_they_claim_to_be():
    """The edge rows exist to make the norm's SIGN observable.

    Every norm a real cache holds is far below 2**15, so a score lane that
    read the norm as unsigned agrees with all 2,080 real rows. These rows have
    to actually sign-extend negative, or the bench they feed proves nothing.
    """
    vs, kernel = _built()
    with tempfile.TemporaryDirectory() as d:
        names = {os.path.basename(p) for p in emit(vs, d)}
        assert {"score_edge_rows.hex", "score_edge_gold.hex",
                "qtab_table.hex", "cb_centroids_key.hex"} <= names
        rows = [ln for ln in open(os.path.join(d, "score_edge_rows.hex")).read().split()]
        assert len(rows) == 5
        buf = np.array([[int(ln[i:i + 2], 16) for i in range(0, len(ln), 2)][::-1]
                        for ln in rows], dtype=np.uint8)
        kv = kernel.cache.quantizer.unpack(buf)
        assert (kv.k_norm < 0).sum() == 2, "two of the five norms must be negative"
        assert kv.k_norm.max() == (1 << 15) - 1 and kv.k_norm.min() == -(1 << 15)
        gold = [int(ln, 16) for ln in
                open(os.path.join(d, "score_edge_gold.hex")).read().split()]
        s, _ = kernel.attn.scores(vs.rotated["q"][0][0], kv)
        sign = 1 << 47
        exact(np.array([(g ^ sign) - sign for g in gold]), np.ravel(s),
              "score_edge_gold.hex")


def check_the_online_observer_leaves_no_trace():
    """`CompressedAttention.on_step` must be detached again, like `on_stage`.

    A vector generator that leaves an observer attached turns every later call
    into a silent memory leak and, worse, makes a second `collect` record into
    the first one's list. The kernel's own tap is checked this way already;
    this is the same contract on the attention object.
    """
    vs, kernel = _built()
    assert kernel.attn.on_step is None, "the online observer must be detached again"
    assert vs.softmax, "the recurrence was never tapped"
    assert vs.softmax["acc"].shape == (vs.context + 1, vs.model.kv_groups,
                                       vs.model.head_dim)


def check_the_real_scores_do_not_exercise_the_rescale():
    """The reason a second, scaled query exists at all.

    At ctx 64 the four lanes take 19 new maxima between them and the rescale
    factor is unity -- 32,768 -- in all but three. Consecutive maxima differ by
    a few LSBs of Q16, so `exp(-delta)` rounds to 1.0 and `acc * factor >> 15`
    returns `acc` unchanged: the multiply is present but numerically inert, and
    a testbench built only on this stimulus cannot tell a correct rescale from
    a rounded one. This check pins that fact, so if the golden data ever starts
    exercising the rescale on its own, the extra stimulus can go.
    """
    vs, _ = _built(ctx=64)          # the context the shipped vectors are built at
    grew = vs.softmax["grew"].astype(bool)
    factors = vs.softmax["factor"][grew]
    unity = 1 << vs.fmt.prob_frac
    assert grew.sum() == 19
    # Four are the zero of each lane's first token, where `m` starts at the
    # format's floor and annihilates an empty accumulator. Thirteen are exactly
    # unity, so the multiply runs and changes nothing. TWO of nineteen events
    # do arithmetic a mutation could get wrong.
    assert (factors == 0).sum() == vs.model.kv_groups
    assert (factors == unity).sum() == 13
    assert len(set(int(x) for x in factors)) == 3


def check_the_stress_query_spreads_the_rescale_factors():
    """The scaled query must produce factors across the range, not near unity.

    The manifest carries the count so a regeneration that quietly stopped
    spreading them fails here rather than in a testbench that still passes
    while testing nothing.
    """
    vs, _ = _built()
    with tempfile.TemporaryDirectory() as d:
        names = {os.path.basename(p) for p in emit(vs, d)}
        assert {"sm2_factor.hex", "sm2_acc.hex", "sm2_q.hex",
                "exp_table.hex", "exp_in.hex", "exp_out.hex"} <= names
        man = json.load(open(os.path.join(d, "manifest.json")))
        assert man["stress_distinct_factors"] >= 8, man["stress_distinct_factors"]

        # The stress query must still fit the seam it claims to cross.
        lim = 1 << (vs.lane_bits - 1)
        q = [int(ln[i:i + 6], 16) for ln in
             open(os.path.join(d, "sm2_q.hex")).read().split()
             for i in range(0, len(ln), 6)]
        assert all(-lim <= (x ^ lim) - lim < lim for x in q)


def check_the_exp_sweep_covers_every_reachable_output():
    """`exp_in.hex` must straddle every step of the LUT, not sample it.

    The hardware splits `delta * log2e` into a table index and a shift. Every
    boundary between two outputs is where an off-by-one in that split lives,
    and uniform random stimulus plateaus over all of them.
    """
    vs, kernel = _built()
    e = kernel.attn.exp
    with tempfile.TemporaryDirectory() as d:
        emit(vs, d)
        dv = np.array([int(x, 16) for x in
                       open(os.path.join(d, "exp_in.hex")).read().split()],
                      dtype=np.int64)
        pv = np.array([int(x, 16) for x in
                       open(os.path.join(d, "exp_out.hex")).read().split()],
                      dtype=np.int64)
        exact(pv, e(dv), "exp_out.hex")
        exact(e(dv), e.two_step(dv), "the flat and two-step exp agree")
        # Every one of the 4,353 reachable outputs is produced by some delta.
        assert len(set(int(x) for x in pv)) == len(set(int(x) for x in e._flat))
