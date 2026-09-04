"""Turn a trace into a performance report.

This is the only place cycles are produced, and it runs strictly after the
values are computed. Nothing here can reach back into an op.

THE COMPOSITION MODEL IS STATED, NOT ASSUMED
--------------------------------------------
The four arrays and the memory are separate units, so the projections can
overlap each other but not their own weight fetch. The report gives both a
serial total (everything in sequence -- the pessimistic bound) and a critical
path (the slowest unit -- the optimistic bound), because a single number would
be hiding a scheduling assumption that has not been designed yet.

WEIGHT TRAFFIC IS CHARGED ONCE, IN THE ARRAYS
---------------------------------------------
Streaming weights appears in the trace twice by construction: as a MEM_READ
(the port moved the bytes) and inside each MATVEC (the array waited for them).
They are the same bytes seen from the two ends, and adding both would
double-count the dominant term of a decode step.

So `SystolicArray.project` owns it -- it is where the stall actually happens
and where the weight/compute bound is decided -- and the memory model is given
only the non-weight records. Weight bytes are still reported, under
`weight_bytes`, so nothing disappears; it is attributed, not dropped.

EVERY UNIT THE TRACE NAMES MUST EXIST IN THE CONFIG
---------------------------------------------------
`MATVEC` has always been looked up by `getattr(hw, w.unit)`, so a projection
against a hardware config that has no such array is an error. Every other op is
now resolved the same way, through `HardwareConfig` fields named after the
`unit` strings the ops record. Before this, `ROTATE`, `QUANTIZE`, `SOFTMAX`,
`ACCUMULATE` and `RESCALE` cost zero cycles and `SCORE` was a flat `sum(w.n)`
with no lanes in it -- which is to say the block this project is building was
the one part of the design the model did not charge for.

Units that genuinely cost the PL nothing are listed in
`HardwareConfig.HOST_UNITS` with the reason. That is the difference between a
zero that was decided and a zero that was never noticed.

THE CACHE HAS TWO BOUNDS AND BOTH ARE REPORTED
----------------------------------------------
`MemoryModel` charges bursts and their latencies. `CacheBandwidth` charges the
measured DDR rate. `memory_cycles` is the max of the two, and
`memory_bound_by` says which -- at long context it is the bandwidth, which is
the wall the whole split at RoPE exists to put every measured byte behind.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..config import HardwareConfig
from ..trace import Op, Trace
from .attn_block import (AttentionUnit, CacheBandwidth, CacheReport,
                         EncoderUnit, RotateUnit, UnitReport)
from .memory import MemoryModel, MemoryReport
from .pe_array import ArrayReport, SystolicArray


@dataclass
class PerformanceReport:
    memory: MemoryReport                                 # KV cache traffic only
    arrays: dict = field(default_factory=dict)           # unit -> ArrayReport
    units: dict = field(default_factory=dict)            # unit -> UnitReport (PL)
    cache: CacheReport | None = None                     # the DDR bandwidth bound
    clock_mhz: float = 100.0
    weight_bytes: int = 0                                # charged inside `arrays`

    @property
    def array_cycles(self) -> int:
        return sum(r.cycles for r in self.arrays.values())

    @property
    def unit_cycles(self) -> int:
        return sum(r.cycles for r in self.units.values())

    @property
    def attention_cycles(self) -> int:
        """Score + softmax + accumulate. Kept as a name because it is the
        headline of a long-context step, but it is now one of `units`."""
        r = self.units.get("attention")
        return r.cycles if r is not None else 0

    @property
    def memory_cycles(self) -> int:
        """The cache's cost: bursts and latency, or raw bandwidth, whichever binds."""
        return max(self.memory.cycles, self.cache.cycles if self.cache else 0)

    @property
    def memory_bound_by(self) -> str:
        if self.cache is not None and self.cache.cycles > self.memory.cycles:
            return f"bandwidth ({self.cache.bound_by})"
        return "bursts"

    @property
    def cache_bytes(self) -> int:
        return self.memory.bytes_read + self.memory.bytes_written

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.cache_bytes

    @property
    def serial_cycles(self) -> int:
        """Everything in sequence. The bound with no overlap engineered."""
        return self.memory_cycles + self.array_cycles + self.unit_cycles

    @property
    def critical_path_cycles(self) -> int:
        """The slowest unit. The bound with perfect overlap.

        The cache stream is one of the candidates rather than an addition to
        them: at long context it is the largest, and a design that overlapped
        everything perfectly would still be waiting on DDR.
        """
        return max([self.memory_cycles,
                    *(r.cycles for r in self.arrays.values()),
                    *(r.cycles for r in self.units.values()), 0])

    def ns_per_token(self, n_tokens: int, *, overlapped: bool = False) -> float:
        cyc = self.critical_path_cycles if overlapped else self.serial_cycles
        return cyc / self.clock_mhz * 1000.0 / max(n_tokens, 1)

    def summary(self, n_tokens: int = 1) -> str:
        share = self.weight_bytes / max(self.total_bytes, 1)
        lines = [
            f"traffic    {self.weight_bytes:,} B weights + "
            f"{self.cache_bytes:,} B KV = {self.total_bytes:,} B "
            f"({share:.0%} weights)",
            f"kv memory  {self.memory}",
        ]
        if self.cache is not None:
            lines.append(f"cache ddr  {self.cache}")
            lines.append(f"kv bound   {self.memory_bound_by} -> "
                         f"{self.memory_cycles:,} cyc")
        for unit, r in sorted(self.arrays.items()):
            lines.append(f"{unit:<10} {r}")
        for unit, r in sorted(self.units.items()):
            lines.append(f"{unit:<10} {r}")
        lines.append(
            f"total      {self.serial_cycles:,} cyc serial / "
            f"{self.critical_path_cycles:,} cyc critical path  "
            f"@ {self.clock_mhz:.0f} MHz")
        lines.append(
            f"per token  {self.ns_per_token(n_tokens):.1f} ns serial / "
            f"{self.ns_per_token(n_tokens, overlapped=True):.1f} ns overlapped")
        return "\n".join(lines)


def replay(trace: Trace, hw: HardwareConfig, n_tokens: int = 1) -> PerformanceReport:
    """Consume a trace against a hardware configuration."""
    # Split weight reads out before the memory model sees them; the arrays
    # charge for that traffic. See the module docstring.
    weight_bytes = sum(w.n_bytes for w in trace.records
                       if w.meta.get("what") == "weights")
    cache_trace = Trace()
    cache_trace.records = [w for w in trace.records
                           if w.meta.get("what") != "weights"]
    mem = MemoryModel(hw.memory).replay(cache_trace)

    arrays: dict[str, ArrayReport] = {}
    grouped: dict[str, list] = defaultdict(list)
    for w in trace.of(Op.MATVEC):
        grouped[w.unit].append(w)

    for unit, works in grouped.items():
        cfg = getattr(hw, unit, None)
        if cfg is None:
            raise ValueError(f"trace names unit {unit!r}, absent from the hardware config")
        model = SystolicArray(cfg, hw.memory)
        reports = [model.project(w.m, w.k, w.n) for w in works]
        arrays[unit] = _sum_reports(reports)

    # The PL blocks, resolved by unit name exactly as MATVEC is. `Op` decides
    # which model reads a record; `unit` decides which instance of it.
    by_unit: dict[str, list] = defaultdict(list)
    for w in trace.records:
        if w.op is Op.MATVEC or w.unit in ("", "memory"):
            continue
        by_unit[w.unit].append(w)

    units: dict[str, UnitReport] = {}
    for unit, works in by_unit.items():
        if unit in HardwareConfig.HOST_UNITS:
            continue                       # runs on the host. See the config.
        cfg = getattr(hw, unit, None)
        if cfg is None:
            raise ValueError(f"trace names unit {unit!r}, absent from the hardware config")
        model = _UNIT_MODELS.get(unit)
        if model is None:
            raise ValueError(f"unit {unit!r} is configured but has no cycle model")
        units[unit] = model(cfg).cost(works)

    # The cache's second bound: can the DRAM sustain the rate at all. Charged
    # on the same bytes the memory model saw, so the two are comparable.
    cache = CacheBandwidth(hw.cache, hw.clock_mhz).cost(
        mem.bytes_read + mem.bytes_written)

    return PerformanceReport(memory=mem, arrays=arrays, units=units, cache=cache,
                             clock_mhz=hw.clock_mhz,
                             weight_bytes=weight_bytes)


# Unit name -> the block that models it. The name is the trace's `unit` string
# and the `HardwareConfig` field, so the three cannot drift apart silently.
_UNIT_MODELS = {
    "rotate": RotateUnit,
    "encoder": EncoderUnit,
    "attention": AttentionUnit,
}


def _sum_reports(rs: list[ArrayReport]) -> ArrayReport:
    if not rs:
        raise ValueError("no reports to sum")
    f = rs[0]
    return ArrayReport(
        cycles=sum(r.cycles for r in rs),
        compute_cycles=sum(r.compute_cycles for r in rs),
        weight_cycles=sum(r.weight_cycles for r in rs),
        fill_cycles=sum(r.fill_cycles for r in rs),
        tiles=sum(r.tiles for r in rs),
        macs=sum(r.macs for r in rs),
        pe_count=f.pe_count,
        weight_bytes=sum(r.weight_bytes for r in rs),
    )
