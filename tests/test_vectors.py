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
