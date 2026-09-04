"""The PL half of the seam, modelled in cycles.

`pe_array.py` models the four systolic arrays, which stay on the HOST. This
models what crosses to the FPGA: rotate -> quantize -> cache -> score ->
online softmax -> accumulate. Same contract as every other model in `kernel.hw`
-- it is handed `Work` records after the values exist, it holds no state, and
it cannot reach back into an op.

WHAT THIS FIXES
---------------
Before this file, `replay` charged `ROTATE`, `QUANTIZE`, `SOFTMAX`,
`ACCUMULATE` and `RESCALE` **zero cycles** and `SCORE` a flat `sum(w.n)` -- one
cycle per (query, cached token) pair with no lanes in it. Every one of those is
a piece of the block this project is actually building, so the report was a
model of the projections with the contribution missing.

THE BOUND THAT MATTERS IS NOT ARITHMETIC
----------------------------------------
The score loop has three DSPs in it (`plan.MD`, "The result that shapes the
datapath"): scores are gathers into a precomputed product table and an adder
tree, and accumulation is a 4:1 mux. So the compute cost of Phase B is one
cached row per cycle per KV head and no amount of logic moves it -- what moves
it is how fast 52 B rows arrive from DDR.

That is why `CacheReport` carries a bandwidth bound beside the burst/latency
model in `memory.py` and takes the MAXIMUM. The two answer different questions:
`MemoryModel` asks what the bursts cost, this asks whether the DRAM can keep
up at all. At short context the first binds; at long context -- the case this
design exists for -- the second does, and it does so from a MEASURED number
(`CacheConfig.ddr_gbps`), not an assumed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import AttentionConfig, CacheConfig, EncoderConfig, RotateConfig
from ..trace import Op, Work


@dataclass(frozen=True)
class UnitReport:
    """Cycles for one PL unit, with the binding constraint left visible.

    `bound_by` is never derived by the caller. A unit that cannot be bound two
    ways reports its own name, so a report can be read without knowing which
    units have a choice.
    """

    unit: str
    cycles: int
    throughput_cycles: int
    latency_cycles: int
    items: int                      # vectors, rows, or (query, token) pairs
    bound_by: str

    def __str__(self) -> str:
        return (f"{self.cycles:,} cyc  bound by {self.bound_by}  "
                f"({self.items:,} items, {self.latency_cycles:,} cyc latency)")


# --------------------------------------------------------------------------
# Phase A -- once per token
# --------------------------------------------------------------------------

class RotateUnit:
    """`rot_fwht.sv`. One vector per cycle, `log2(d)` stages deep.

    A `ROTATE` record carries `m=d`, `k=` how many vectors, `n=` how many
    rounds, and `log2d` in its meta. Rounds are sequential passes over the same
    butterfly, so they multiply the vector count rather than the depth.
    """

    def __init__(self, cfg: RotateConfig) -> None:
        self.cfg = cfg

    def cost(self, works: list[Work]) -> UnitReport:
        vectors = sum(max(w.k, 1) * max(w.n, 1) for w in works)
        depth = max((int(w.meta.get("log2d", 0)) for w in works), default=0)
        latency = depth * self.cfg.stage_latency
        thru = math.ceil(vectors / self.cfg.vectors_per_cycle)
        return UnitReport("rotate", thru + latency, thru, latency, vectors, "fwht")


class EncoderUnit:
    """`rot_norm.sv` + `rot_encode.sv`. One rotated vector in, codes out.

    A `QUANTIZE` record counts `n = 2 * n_tokens` vectors -- a key plane and a
    value plane -- which is exactly what the hardware pushes through the norm
    and the threshold compare.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg

    def cost(self, works: list[Work]) -> UnitReport:
        vectors = sum(max(w.n, 1) for w in works)
        thru = math.ceil(vectors / self.cfg.vectors_per_cycle)
        latency = self.cfg.norm_latency
        return UnitReport("encoder", thru + latency, thru, latency, vectors, "encode")


# --------------------------------------------------------------------------
# Phase B -- once per cached token. >99% of a long-context step.
# --------------------------------------------------------------------------

class AttentionUnit:
    """Score, softmax and accumulate, over `lanes` query heads at a time.

    SCORE and ACCUMULATE both count (query, cached token) pairs, and they are
    the same pairs seen at the two ends of one pipeline -- a scored row is
    accumulated in the same cycle it is scored, one stage later. Charging both
    would double the dominant term of a long-context step for the same reason
    `replay` refuses to charge weight bytes twice. So the unit takes the
    MAXIMUM of the two, not the sum, and SOFTMAX rides along inside it.

    RESCALE is the exception and is charged on top: a running-maximum update
    stalls the pipe, and the whole reason the online form is modelled at all is
    to find out how often that happens.
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        self.cfg = cfg

    def cost(self, works: list[Work]) -> UnitReport:
        pairs = max(
            (sum(max(w.n, 1) for w in works if w.op is op) for op in
             (Op.SCORE, Op.ACCUMULATE)),
            default=0,
        )
        rescales = sum(max(w.n, 0) for w in works if w.op is Op.RESCALE)
        thru = math.ceil(pairs / self.cfg.lanes) + rescales * self.cfg.rescale_cycles
        latency = self.cfg.score_latency
        bound = "rescale" if rescales * self.cfg.rescale_cycles > thru // 2 else "scan"
        return UnitReport("attention", thru + latency, thru, latency, pairs, bound)


# --------------------------------------------------------------------------
# The wall
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CacheReport:
    """The DDR bandwidth bound on the cache stream.

    Separate from `MemoryReport` rather than folded into it, because they are
    two different claims about the same bytes and both are worth reading: one
    says what the bursts and their latencies cost, the other says whether the
    DRAM can sustain the rate at all.
    """

    cycles: int
    bytes_moved: int
    bytes_per_cycle: float
    bound_by: str                   # "interface" (AXI-HP ports) or "dram"

    @property
    def gbps(self) -> float:
        return self.bytes_per_cycle

    def __str__(self) -> str:
        return (f"{self.cycles:,} cyc  {self.bytes_moved:,} B at "
                f"{self.bytes_per_cycle:.1f} B/cyc  limited by {self.bound_by}")


class CacheBandwidth:
    """`kv_store_ddr.sv`. Bytes in, cycles out, at a measured rate."""

    def __init__(self, cfg: CacheConfig, clock_mhz: float) -> None:
        self.cfg = cfg
        self.clock_mhz = clock_mhz

    def cost(self, n_bytes: int) -> CacheReport:
        bpc = self.cfg.bytes_per_cycle(self.clock_mhz)
        if bpc <= 0:
            raise ValueError(f"cache bandwidth is {bpc} B/cycle; nothing can stream")
        return CacheReport(
            cycles=math.ceil(n_bytes / bpc),
            bytes_moved=n_bytes,
            bytes_per_cycle=bpc,
            bound_by=self.cfg.bound_by(self.clock_mhz),
        )
