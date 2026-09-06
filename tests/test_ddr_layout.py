"""The DDR layout must be a regrouping of `pack`'s bytes and nothing else.

`kv_store_ddr.sv` reads four planes and hands `score_lane` a 52-byte row. If the
transpose here and the reassembly there disagree by one byte, every cached token
decodes into plausible garbage of exactly the right size -- the same failure mode
`_pack_codes` has, and the reason it is checked the same way.
"""

import numpy as np

from kernel.hw.ddr_layout import BEAT_BYTES, DdrLayout
from kernel.hw.vectors import build
from kernel.tests.harness import exact


def _layout(vs, kernel):
    return DdrLayout.for_quant(kernel.cfg.quant, vs.model.head_dim,
                               vs.model.num_kv_heads, vs.context + 1)


def check_every_row_survives_the_transpose():
    """Byte-exact, every token, every head. The only property anything needs."""
    vs, kernel = build(ctx=32)
    lay = _layout(vs, kernel)
    img = lay.image(vs.cache)
    for h in range(vs.model.num_kv_heads):
        for t in range(vs.context + 1):
            exact(lay.row_at(img, h, t), vs.cache[t, h],
                  f"head {h} token {t} round trip")


def check_the_reassembled_row_still_unpacks_to_the_cache():
    """Not just the same bytes -- the same codes and norms.

    A transpose that was self-consistent but disagreed with `pack` would pass
    the byte round trip above if `image` and `row_at` shared the mistake. This
    goes through `KVQuantizer.unpack` instead, which knows nothing about either.
    """
    vs, kernel = build(ctx=32)
    lay = _layout(vs, kernel)
    img = lay.image(vs.cache)
    qz = kernel.cache.quantizer
    for h in (0, vs.model.num_kv_heads - 1):
        rows = np.stack([lay.row_at(img, h, t) for t in range(vs.context + 1)])
        got = qz.unpack(rows)
        want = kernel.cache.view(h)
        exact(got.k_idx, want.k_idx, f"head {h} k codes")
        exact(got.v_idx, want.v_idx, f"head {h} v codes")
        exact(got.k_norm, want.k_norm, f"head {h} k norms")
        exact(got.v_norm, want.v_norm, f"head {h} v norms")


def check_the_port_count_is_derived_and_not_chosen():
    """The plan derived 4 ports from bandwidth; the layout derives 4 from the row.

    52 B/cycle against a 16 B/cycle port needs 4 ports. Splitting a 52-byte row
    at its fields and then at beats also gives 4. Two independent arguments, and
    they have to agree or one of them is wrong.
    """
    from kernel.config import QuantConfig

    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8, 64)
    assert lay.n_ports == 4 == -(-52 // BEAT_BYTES)
    assert [p.width for p in lay.planes] == [16, 16, 16, 4]
    assert sum(p.width for p in lay.planes) == lay.row_bytes == 52
    edges = [(p.lo, p.hi) for p in lay.planes]
    assert edges[0][0] == 0 and edges[-1][1] == 52
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))


def check_every_quantisation_width_gives_beat_friendly_planes():
    """All 64 (key_bits, value_bits) pairs step 15 can sweep.

    Chopping the row blindly every 16 bytes is simpler and breaks on half of
    them: `kb + vb = 3` gives a 28-byte row and a trailing plane of 12, and 12
    neither divides a 16-byte beat nor is a multiple of one. Splitting at fields
    first cannot produce that, because every field is 8*kb, 8*vb or 4 bytes.
    """
    from kernel.config import QuantConfig

    blind_failures = 0
    for kb in range(1, 9):
        for vb in range(1, 9):
            q = QuantConfig(key_bits=kb, value_bits=vb)
            lay = DdrLayout.for_quant(q, 64, 8, 64)
            for pl in lay.planes:
                assert pl.width <= BEAT_BYTES and BEAT_BYTES % pl.width == 0, \
                    (kb, vb, pl.width)
            assert sum(p.width for p in lay.planes) == lay.row_bytes
            assert lay.bytes_per_cycle == lay.row_bytes

            # The blind split this replaced, for the record.
            rb = lay.row_bytes
            widths = [min(BEAT_BYTES, rb - BEAT_BYTES * i)
                      for i in range(-(-rb // BEAT_BYTES))]
            if any(BEAT_BYTES % w for w in widths):
                blind_failures += 1
    assert blind_failures == 32, blind_failures


def check_each_port_reads_a_whole_number_of_tokens_per_beat():
    """The property that makes the transpose worth doing.

    Every plane is either a whole number of beats per token or a whole number
    of tokens per beat. Neither needs a byte shift register, which is what a
    flat 52-byte row would have forced on all four ports.
    """
    from kernel.config import QuantConfig
    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8, 64)
    for p in lay.planes:
        assert p.width % BEAT_BYTES == 0 or BEAT_BYTES % p.width == 0, p
    # And the four together are exactly one row per cycle.
    assert sum(p.width for p in lay.planes) == lay.bytes_per_cycle == 52


def check_planes_are_page_aligned_and_do_not_overlap():
    """A burst must never straddle a page boundary it did not have to."""
    from kernel.config import QuantConfig
    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8, 4096)
    seen = []
    for h in range(lay.n_kv_heads):
        for p in range(lay.n_ports):
            a = lay.plane_base(h, p)
            assert a % 4096 == 0, (h, p, a)
            seen.append((a, a + lay.capacity * lay.planes[p].width))
    seen.sort()
    for (a0, a1), (b0, _) in zip(seen, seen[1:]):
        assert a1 <= b0, f"planes overlap: {a1} > {b0}"
    assert seen[-1][1] <= lay.base + lay.total_bytes


def check_the_norm_plane_is_not_padded_to_a_beat():
    """Padding it would push the engine to the interface ceiling exactly.

    The norms are 4 B/token. Storing them one per 16 B beat would make that
    port read 16 B/token like the others, taking the engine from 52 B/cycle to
    64 -- which is the whole four-port interface and leaves the DRAM controller
    no headroom at all against a measured 14.7 GB/s.
    """
    from kernel.config import QuantConfig
    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8, 4096)
    norms = lay.planes[-1]
    assert norms.width == 4
    # The property is the stride within the plane, not the plane's total span:
    # consecutive tokens are 4 B apart, so the port reads 4 B/token and the
    # engine stays at 52 B/cycle. The span itself is padded to the widest
    # plane's, which the hardware requires and which no port ever reads into --
    # see `DdrLayout.plane_span` and `check_every_plane_has_the_same_span`.
    for t in range(4):
        assert lay.token_addr(0, 3, t + 1) - lay.token_addr(0, 3, t) == 4
    assert lay.bytes_per_cycle == 52
    padded = sum(BEAT_BYTES for _ in lay.planes)
    assert padded == 64, padded


def check_every_plane_has_the_same_span():
    """`attn_top.sv` walks the planes as `cache_base + p * plane_span`.

    One span register, because per-plane bases put a 49-bit multiply on the
    path that missed 250 MHz by 1.953 ns. So a layout whose four planes have
    four different spans is one the hardware cannot address -- and at capacity
    65, where every plane rounds to a single page anyway, that is invisible.
    Every vector set in this project is built at capacity 65. This is the check
    that runs at the sizes step 13 actually loads.
    """
    from kernel.config import QuantConfig
    q = QuantConfig(key_bits=4, value_bits=2)
    for cap in (65, 257, 1025, 2049, 4097, 32769):
        lay = DdrLayout.for_quant(q, 64, 8, cap)
        spans = [lay.plane_span(p) for p in range(lay.n_ports)]
        assert len(set(spans)) == 1, (cap, spans)
        # And the one span really is what the RTL's arithmetic produces.
        for h in range(lay.n_kv_heads):
            for p in range(lay.n_ports):
                assert lay.plane_base(h, p) == (
                    lay.base + h * spans[0] * lay.n_ports + p * spans[0]), (cap, h, p)
        # Padding is only ever after the last token, so no read reaches it.
        assert spans[0] >= cap * max(pl.width for pl in lay.planes)


def check_the_span_is_the_narrowest_that_holds_the_widest_plane():
    """Padding costs address space, and it must not cost more than it has to.

    A span rounded to the widest plane's need and no further: at capacity
    32,769 that is 16.9 MB against the 13.8 MB a four-span map would take, in a
    2 GB DDR. A span rounded to, say, the next power of two would be 33.6 MB
    for nothing.
    """
    from kernel.config import QuantConfig
    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8, 32769)
    widest = max(pl.width for pl in lay.planes)
    need = 32769 * widest
    assert lay.plane_span(0) - need < 4096, (lay.plane_span(0), need)
