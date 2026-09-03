"""The timing models, and the invariant that they are observers only."""

import numpy as np

from kernel import (ArrayConfig, HardwareConfig, KernelConfig, MemoryConfig,
                    ModelConfig, Op, Trace)
from kernel.hw import MemoryModel, SystolicArray, replay
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
