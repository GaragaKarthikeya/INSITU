"""The board's vectors must be the bench's vectors, in the shim's word order.

`sw/attn_vectors.c` is what the A53 compares against, and it reaches the board
through a path -- numpy to C array to a 32-bit AXI write -- that the twelve
testbenches never exercise. The one thing that path can silently get wrong is
the word order: `hw/vectors.py` emits MSB-first hex beats for `$readmemh` and
`board_vectors` emits little-endian 32-bit words for a memcpy, and a mistake
between the two scrambles the sixteen words of every beat into something that
looks exactly like a datapath that permutes channels.

So the two are checked against each other here, at the one context where both
exist.
"""

import numpy as np

from kernel.hw import board_vectors as bv
from kernel.hw.vectors import build, pack_lanes
from kernel.tests.harness import exact


def _words_from_beats(lines, per_beat=16):
    """MSB-first hex beats -> the 32-bit words at ascending addresses.

    The low word of a beat is the last eight hex digits of the line, and it
    sits at the lowest address. This is `scripts/jtag_attn.tcl`'s reader,
    rewritten in Python -- deliberately, because agreeing with the generator is
    not evidence and agreeing with the thing that already ran on hardware is.
    """
    out = []
    for line in lines:
        h = line.strip().rjust(per_beat * 8, "0")
        for w in range(per_beat - 1, -1, -1):
            out.append(int(h[w * 8:w * 8 + 8], 16))
    return np.array(out, dtype="<u4")


def check_the_board_ingress_is_the_benchs_ingress():
    """Same seed, same context, same 2,304 words."""
    vs, _ = build(ctx=64)
    want = _words_from_beats(vs.ingress_beats())
    case = bv.build_case(64, 1)
    exact(case["ingress"], want, "ingress words")
    assert want.size == bv.IN_WORDS


def check_the_board_golden_is_the_benchs_golden():
    """`out.online.hex` and the C array are the same 1,536 words."""
    vs, _ = build(ctx=64)
    from kernel.hw.vectors import beats
    want = _words_from_beats(beats(pack_lanes(vs.out_online, vs.lane_bits)))
    case = bv.build_case(64, 1)
    exact(case["golden"], want, "golden words")
    assert want.size == bv.OUT_WORDS


def check_the_image_holds_the_old_tokens_and_none_of_the_new_ones():
    """The property the multi-step run is for.

    Every token from `ctx0` on must be zero in the loaded image, or a block
    whose write master did nothing would still answer correctly and the run
    would prove only that reads work.
    """
    from kernel.config import QuantConfig
    from kernel.hw.ddr_layout import DdrLayout

    case = bv.build_case(32, 3)
    lay = DdrLayout.for_quant(QuantConfig(key_bits=4, value_bits=2), 64, 8,
                              case["capacity"])
    nonzero = 0
    for h in range(8):
        for t in range(case["ctx0"]):
            if lay.row_at(case["image"], h, t).any():
                nonzero += 1
        for t in range(case["ctx0"], case["capacity"]):
            assert not lay.row_at(case["image"], h, t).any(), (h, t)
    # Not a vacuous check: the old tokens really are there.
    assert nonzero > 8 * case["ctx0"] * 0.9, nonzero


def check_each_step_scores_over_one_more_token():
    """`n_tokens` is `ctx+1` and it advances by one, which is the whole point.

    If it did not, every step would read the same rows and the run would say
    nothing about whether the previous step's write landed.
    """
    case = bv.build_case(64, 4)
    exact(np.asarray(case["n_tokens"], dtype=np.int64),
          np.array([65, 66, 67, 68]), "n_tokens")
    assert case["capacity"] == 68


def check_the_steps_are_not_all_the_same_answer():
    """Four identical goldens would pass a broken run four times."""
    case = bv.build_case(64, 4)
    g = case["golden"].reshape(4, bv.OUT_WORDS)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(g[i], g[j]), (i, j)


def check_the_plane_span_is_one_number_the_hardware_can_use():
    """`attn_top` has one `plane_span` register; the case must hand it one."""
    for ctx0, steps in ((64, 4), (256, 4), (1024, 4)):
        case = bv.build_case(ctx0, steps)
        assert case["head_stride"] == 4 * case["plane_span"], case
        assert case["image"].size == case["n_kv_heads"] * case["head_stride"]
        assert case["plane_span"] >= case["capacity"] * 16
