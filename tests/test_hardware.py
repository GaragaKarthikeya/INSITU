"""The timing models, and the invariant that they are observers only."""

import functools

import numpy as np

from kernel import (ArrayConfig, AttentionConfig, CacheConfig, HardwareConfig,
                    KernelConfig, MemoryConfig, ModelConfig, Op, QuantConfig,
                    Trace)
from kernel.hw import AttentionUnit, CacheBandwidth, MemoryModel, SystolicArray, replay
from kernel import AttentionKernel
from .harness import raises

MEM = MemoryConfig(port_bits=256, read_latency=100, write_latency=8)


def check_gemv_is_weight_bound_and_no_array_size_fixes_it():
    """Decode's central performance fact, asserted rather than described.

    A matrix-vector product uses every weight exactly once, so the array is
    starved no matter how large it is. Growing it 16x must not help.
    """
    small = SystolicArray(ArrayConfig(rows=32, cols=32), MEM).project(2048, 2048, 1)
    big = SystolicArray(ArrayConfig(rows=128, cols=128), MEM).project(2048, 2048, 1)
    assert small.bound_by == big.bound_by == "weights"
    assert big.utilization < 0.02, f"util {big.utilization:.2%} unexpectedly high"
    # 16x the PEs buys well under 2x, and only from the fill amortising.
    assert small.cycles / big.cycles < 2.0


def check_batching_is_what_moves_the_bound():
    """Prefill reuses each weight once per token, so the array can fill up.

    The crossover is not a matter of taste. A tile holds `rows*cols` weights;
    at `port_bits/8` bytes per beat and 2 B per weight the port delivers
    `port_bits/16` weights per cycle, so a tile takes
    `rows*cols / (port_bits/16)` cycles to load and the array is weight-bound
    until the batch exceeds that. For 128x128 on a 256-bit port that is 1024
    tokens -- which is far more batch than decode will ever have, and is the
    real reason widening the port beats widening the array.
    """
    cfg = ArrayConfig(rows=128, cols=128)
    a = SystolicArray(cfg, MEM)
    crossover = cfg.rows * cfg.cols // (MEM.port_bits // 16)
    assert crossover == 1024

    assert a.project(2048, 2048, 1).bound_by == "weights"
    assert a.project(2048, 2048, crossover - 1).bound_by == "weights"
    assert a.project(2048, 2048, crossover + 1).bound_by == "compute"
    assert a.project(2048, 2048, 4096).utilization > \
           20 * a.project(2048, 2048, 1).utilization


def check_a_wider_port_moves_the_crossover_not_the_array():
    """Doubling the port halves the batch needed to saturate; doubling the array does not."""
    cfg = ArrayConfig(rows=128, cols=128)
    wide = MemoryConfig(port_bits=512, read_latency=100)
    assert SystolicArray(cfg, wide).project(2048, 2048, 600).bound_by == "compute"
    assert SystolicArray(cfg, MEM).project(2048, 2048, 600).bound_by == "weights"


def check_array_rejects_degenerate_shapes():
    a = SystolicArray(ArrayConfig(), MEM)
    raises(ValueError, lambda: a.project(0, 16, 1), "m=0")


def check_memory_charges_latency_per_burst():
    t = Trace()
    t.add(Op.MEM_READ, "memory", n_bytes=32)
    t.add(Op.MEM_READ, "memory", n_bytes=32)
    one = Trace(); one.add(Op.MEM_READ, "memory", n_bytes=64)
    m = MemoryModel(MEM)
    assert m.replay(t).cycles > m.replay(one).cycles, "two bursts must cost two latencies"
    assert m.replay(t).bursts == 2 and m.replay(one).bursts == 1


def check_beat_padding_is_counted():
    """44 B on a 256-bit port occupies 64 B. The 20 B is charged, not ignored."""
    t = Trace()
    t.add(Op.MEM_WRITE, "memory", n_bytes=44)
    r = MemoryModel(MEM).replay(t)
    assert r.padding_bytes == 20
    assert r.efficiency < 0.7


def check_narrower_port_is_more_byte_efficient_for_this_record_size():
    """The counter-intuitive result the padding model exists to expose."""
    t = Trace()
    for _ in range(100):
        t.add(Op.MEM_READ, "memory", n_bytes=44)
    wide = MemoryModel(MemoryConfig(port_bits=256)).replay(t)
    narrow = MemoryModel(MemoryConfig(port_bits=64)).replay(t)
    assert narrow.efficiency > wide.efficiency
    assert narrow.padding_bytes < wide.padding_bytes


def check_beat_gap_and_latency_are_distinguishable():
    t = Trace()
    t.add(Op.MEM_READ, "memory", n_bytes=1 << 16)
    lat = MemoryModel(MemoryConfig(read_latency=1000, beat_gap=0)).replay(t).cycles
    gap = MemoryModel(MemoryConfig(read_latency=0, beat_gap=1)).replay(t).cycles
    assert lat != gap, "the two knobs must not collapse into one"


def check_replay_rejects_an_unknown_unit():
    t = Trace()
    t.add(Op.MATVEC, "nonexistent_array", m=16, k=16, n=1)
    raises(ValueError, lambda: replay(t, HardwareConfig()), "unknown unit")


def check_replay_produces_both_bounds():
    m = ModelConfig(hidden_size=128, num_heads=4, num_kv_heads=2, head_dim=32)
    hw = HardwareConfig.sized_for(m, side=32)
    from kernel import AttentionKernel
    k = AttentionKernel(KernelConfig(model=m, hw=hw), capacity=64)
    t = Trace()
    k.forward(np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32),
              trace=t)
    r = replay(t, hw, n_tokens=4)
    # Weight traffic is attributed to the arrays, not to the memory model, so
    # the dominant term of a decode step is not counted twice.
    assert r.weight_bytes > 0
    assert r.memory.bytes_read < r.weight_bytes
    assert r.total_bytes == r.weight_bytes + r.cache_bytes
    assert r.critical_path_cycles <= r.serial_cycles
    assert r.ns_per_token(4, overlapped=True) <= r.ns_per_token(4)
    assert set(r.arrays) == {"q_array", "k_array", "v_array", "o_array"}
    # Every PL block the trace names is charged. Before the cost model, all of
    # these except SCORE cost zero cycles -- the block being built was the one
    # part of the design the report left out.
    assert set(r.units) == {"rotate", "encoder", "attention"}
    assert all(u.cycles > 0 for u in r.units.values())
    assert r.cache is not None and r.cache.cycles > 0


# --------------------------------------------------------------------------
# the PL blocks
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _zcu104(ctx: int, n_query: int = 1):
    """One decode step against a `ctx`-token cache, on the real board config.

    Cached: the prefill that builds the cache is the expensive part and none of
    the checks below mutate what comes back.
    """
    m = ModelConfig(hidden_size=2048, num_heads=32, num_kv_heads=8, head_dim=64,
                    max_position=1 << 18)
    hw = HardwareConfig.zcu104(m)
    cfg = KernelConfig(model=m, hw=hw,
                       quant=QuantConfig(key_bits=4, value_bits=2))
    k = AttentionKernel(cfg, capacity=ctx + n_query)
    rng = np.random.default_rng(0)
    if ctx:
        k.forward(rng.standard_normal((ctx, 2048)).astype(np.float32) * 0.02)
    t = Trace()
    k.forward(rng.standard_normal((n_query, 2048)).astype(np.float32) * 0.02, trace=t)
    return replay(t, hw, n_tokens=n_query), hw, k


def check_ddr_bandwidth_is_the_reported_bound_at_long_context():
    """The exit criterion of step 2, and the claim the whole split rests on.

    At a long context the cache scan is not burst-limited or latency-limited --
    it is limited by how fast DDR4 delivers bytes, at the rate MEASURED in step
    1. If this ever comes back "bursts", the model has stopped describing the
    machine the plan sizes: the burst model would be charging for a narrower
    port than the four AXI-HP masters the 14.7 GB/s was measured on.
    """
    r, hw, _ = _zcu104(ctx=4096)
    assert r.memory_bound_by.startswith("bandwidth"), r.memory_bound_by
    assert r.cache.bound_by == "dram", \
        "4 x 128 b = 64 B/cyc of interface against 58.8 B/cyc of DRAM: DRAM binds"
    assert r.memory_cycles == r.cache.cycles

    # The measured rate, in this clock's cycles. 14.708 GB/s at 250 MHz.
    assert abs(r.cache.bytes_per_cycle - 58.83) < 0.05
    # And it really is the bytes the memory model saw -- not a separate count.
    assert r.cache.bytes_moved == r.memory.bytes_read + r.memory.bytes_written


def check_the_two_cache_bounds_cross_over():
    """Short context is burst/latency bound; long context is bandwidth bound.

    Both bounds are reported because they have different fixes. If one number
    were reported instead, the crossover -- and therefore which fix applies --
    would be invisible.
    """
    short, _, _ = _zcu104(ctx=8)
    long_, _, _ = _zcu104(ctx=4096)
    assert short.memory_bound_by == "bursts"
    assert long_.memory_bound_by.startswith("bandwidth")


def check_score_lanes_are_the_gqa_win_and_bandwidth_is_not_affected():
    """Four query heads share one cached row, so lanes buy cycles for free.

    This is the one axis in the datapath that parallelises without more DRAM:
    `plan.MD`, "The four axes". Widening it must cut attention cycles and leave
    the cache stream untouched.
    """
    m = ModelConfig(hidden_size=2048, num_heads=32, num_kv_heads=8, head_dim=64)
    assert HardwareConfig.zcu104(m).attention.lanes == m.kv_groups == 4

    t = Trace()
    t.add(Op.SCORE, "attention", m=64, n=4 * 4096)
    t.add(Op.ACCUMULATE, "attention", m=64, n=4 * 4096)
    one = AttentionUnit(AttentionConfig(lanes=1)).cost(t.records)
    four = AttentionUnit(AttentionConfig(lanes=4)).cost(t.records)
    assert one.throughput_cycles == 4 * four.throughput_cycles == 16384


def check_score_and_accumulate_are_not_charged_twice():
    """They are the same pairs at two ends of one pipeline, one stage apart.

    Summing them would double the dominant term of a long-context step, for the
    same reason `replay` refuses to charge weight bytes to both the arrays and
    the memory model.
    """
    t = Trace()
    t.add(Op.SCORE, "attention", m=64, n=1024)
    t.add(Op.ACCUMULATE, "attention", m=64, n=1024)
    t.add(Op.SOFTMAX, "attention", m=1, n=1024)
    r = AttentionUnit(AttentionConfig(lanes=1)).cost(t.records)
    assert r.throughput_cycles == 1024, "not 2048, and not 3072"


def check_a_rescale_costs_cycles():
    """The online softmax's stall is charged, so its cost can be measured."""
    base = Trace(); base.add(Op.SCORE, "attention", m=64, n=1024)
    with_r = Trace()
    with_r.add(Op.SCORE, "attention", m=64, n=1024)
    with_r.add(Op.RESCALE, "attention", m=64, n=50)
    cfg = AttentionConfig(lanes=1, rescale_cycles=1)
    assert AttentionUnit(cfg).cost(with_r.records).cycles == \
           AttentionUnit(cfg).cost(base.records).cycles + 50


def check_phase_b_is_the_whole_step_at_long_context():
    """`plan.MD`: "At ctx 32,768 Phase B is 262,144 cycles against ~200 for A
    and C. Optimise nothing but Phase B." Asserted, not described."""
    r, _, _ = _zcu104(ctx=4096)
    a = r.units["rotate"].cycles + r.units["encoder"].cycles
    # 293 cycles, of which the encoder is 239: step 6's RTL folds 64 channels
    # over 8 lanes, so Phase A costs more than the first estimate here and
    # still disappears next to Phase B. The ratio is the claim, not the 78
    # cycles this originally asserted.
    assert a < 400
    assert r.units["attention"].cycles > 100 * a
    # 8 KV heads x 4097 cached tokens, four query heads per pass.
    assert r.units["attention"].throughput_cycles == 8 * 4097


def check_host_units_are_a_decision_not_an_oversight():
    """RoPE is on the host, so it costs the PL nothing -- deliberately.

    The distinction this pins: a unit absent from the config raises, a unit
    listed in HOST_UNITS is free. Silence is not one of the options.
    """
    assert "rope" in HardwareConfig.HOST_UNITS
    r, _, _ = _zcu104(ctx=8)
    assert "rope" not in r.units

    t = Trace()
    t.add(Op.ROTATE, "nowhere", m=64, k=1, n=1)
    raises(ValueError, lambda: replay(t, HardwareConfig()), "unknown PL unit")


def check_four_ports_are_required_and_three_are_not_enough():
    """`plan.MD`'s port count, derived rather than quoted.

    One 128-bit HP port is 16 B per 250 MHz cycle = 4.0 GB/s. The engine wants
    52 B/cycle, so three ports (48 B/cyc) cannot feed it and four (64 B/cyc)
    can -- and at three the bound moves from the DRAM to the interface, which
    is a different fix.
    """
    three, four = CacheConfig(hp_ports=3), CacheConfig(hp_ports=4)
    assert three.interface_bytes_per_cycle == 48.0
    assert four.interface_bytes_per_cycle == 64.0
    assert three.bound_by(250.0) == "interface"
    assert four.bound_by(250.0) == "dram"
    assert three.bytes_per_cycle(250.0) < 52.0 <= four.bytes_per_cycle(250.0)


def check_the_ddr_measurement_is_the_number_in_the_plan():
    """14.708 GB/s, measured, 4 ports / burst 64 / outstanding 2.

    Pinned as a test because it is the one figure in the whole model that came
    off hardware, and every throughput claim is sized from it. If it changes,
    it must change with a new measurement and this line.
    """
    assert CacheConfig().ddr_gbps == 14.708
    assert CacheConfig().hp_ports == 4 and CacheConfig().parallel_kv == 1
    # 52 B/cycle of appetite against the measured supply: 13% headroom.
    supply = CacheConfig().bytes_per_cycle(250.0)
    assert 0.12 < supply / 52.0 - 1.0 < 0.14


def check_cache_bandwidth_rejects_a_dead_configuration():
    raises(ValueError, lambda: CacheBandwidth(CacheConfig(ddr_gbps=0.0), 250.0).cost(64),
           "zero bandwidth")


# --------------------------------------------------------------------------
# the scan model, against the board
# --------------------------------------------------------------------------

# (ctx+1, scan cycles per step, starve cycles per step) at 8 KV heads, 52-byte
# rows, 250 MHz. `results/step14_dma_f250.log` for the first four -- bit-exact
# runs -- and `results/step14_longctx_probe.log` for 16k and 32k, which are
# address-pattern probes with a PRNG cache and no checked answer.
BOARD_SCAN = [
    (65, 2059, 64),
    (257, 3805, 89),
    (1025, 10295, 77),
    (8193, 76666, 1326),
    (16385, 151168, 2422),
    (32769, 300760, 4809),
]


def check_the_scan_model_reproduces_the_board():
    """`config.py` must predict what the hardware did, or step 15 is unassessable.

    Step 15's exit criterion is "measured bytes, cycles and throughput move as
    `config.py` predicts". Before this the model answered 58.8 B/cycle where the
    board sustains 45.6 -- it took the minimum of the DDR supply and the
    interface, and had no term for the fact that `PARALLEL_KV = 1` retires one
    row per cycle. It was 29% optimistic, which is exactly how `plan.MD`'s
    throughput table came to be 1.15-1.25x optimistic.
    """
    from kernel.config import CacheConfig

    c = CacheConfig()
    errs = {}
    for tokens, cycles, _starve in BOARD_SCAN:
        got = c.scan_cycles(8, tokens)
        errs[tokens] = abs(got - cycles) / cycles
        assert errs[tokens] < 0.06, (tokens, got, cycles, errs[tokens])
    # Four of the six land inside 0.5%. ctx 1,025 is the worst at 5.1%, and it
    # is worst for a reason worth keeping rather than fitting away: `starve_per_
    # row` is one constant set to the SATURATED rate, 0.0183, which is right
    # from ctx 8k out and overstates ctx 1,025's measured 0.0094. A second
    # parameter would close it and would be four numbers fitted to six points.
    # The long contexts are the ones the design is judged at, so the constant is
    # chosen to be right there.
    assert errs[32769] < 0.01, errs
    assert errs[8193] < 0.01, errs
    assert max(errs.values()) < 0.06, errs


def check_the_engine_and_not_the_dram_is_what_bounds_the_scan():
    """The term the model was missing, stated as the property it implies.

    One row per cycle at 52 B and 250 MHz is 13.0 GB/s. The DRAM delivers 14.7
    and the four ports carry 16.0, so neither of them is the limit and no
    amount of either makes the scan faster.
    """
    from kernel.config import CacheConfig

    c = CacheConfig()
    assert c.bound_by(250.0, row_bytes=52) == "engine"
    assert abs(c.bytes_per_cycle(250.0, row_bytes=52) - 52.0) < 1e-9
    # Without the row width the old answer stands, and it is the DRAM figure --
    # kept so a caller that genuinely wants the supply still gets it.
    assert c.bound_by(250.0) == "dram"
    assert abs(c.bytes_per_cycle(250.0) - 58.83) < 0.05
    # A narrower row moves the bound: at 3b/2b the row is 44 B and the engine
    # is still what binds, which is the claim step 15's sweep rests on.
    assert c.bound_by(250.0, row_bytes=44) == "engine"


def check_the_predicted_throughput_matches_the_board_at_long_context():
    """The number the project is judged on, end to end, from the model alone.

    16 layers of Llama 3.2 1B at ctx 32,768: the board did 1,203 us a layer and
    52 tok/s. If the model cannot reproduce that it cannot be used to choose a
    cache format, which is the whole of step 15.
    """
    from kernel.config import CacheConfig

    c = CacheConfig()
    for tokens, measured_us, measured_toks in ((8193, 307, 204),
                                               (32769, 1203, 52)):
        us = c.scan_cycles(8, tokens) / 250.0
        assert abs(us - measured_us) / measured_us < 0.05, (tokens, us, measured_us)
        toks = 1e6 / (16 * us)
        assert abs(toks - measured_toks) / measured_toks < 0.06, (toks, measured_toks)
