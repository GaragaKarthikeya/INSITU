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
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..config import HardwareConfig
from ..trace import Op, Trace
from .memory import MemoryModel, MemoryReport
from .pe_array import ArrayReport, SystolicArray


@dataclass
class PerformanceReport:
    memory: MemoryReport                                 # KV cache traffic only
    arrays: dict = field(default_factory=dict)           # unit -> ArrayReport
    attention_cycles: int = 0
    clock_mhz: float = 100.0
    weight_bytes: int = 0                                # charged inside `arrays`

    @property
    def array_cycles(self) -> int:
        return sum(r.cycles for r in self.arrays.values())

    @property
    def cache_bytes(self) -> int:
        return self.memory.bytes_read + self.memory.bytes_written

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.cache_bytes

    @property
    def serial_cycles(self) -> int:
        """Everything in sequence. The bound with no overlap engineered."""
        return self.memory.cycles + self.array_cycles + self.attention_cycles

    @property
    def critical_path_cycles(self) -> int:
        """The slowest unit. The bound with perfect overlap."""
        return max([self.memory.cycles, self.attention_cycles,
                    *(r.cycles for r in self.arrays.values()), 0])

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
        for unit, r in sorted(self.arrays.items()):
            lines.append(f"{unit:<10} {r}")
        lines.append(f"attention  {self.attention_cycles:,} cyc")
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

    # The compressed attention datapath consumes one cached token per cycle:
    # a score is a d-wide dot product with a codebook lookup, which is the
    # shape the codes were chosen to make cheap. One cycle per token per head.
    attention_cycles = sum(w.n for w in trace.of(Op.SCORE))

    return PerformanceReport(memory=mem, arrays=arrays,
                             attention_cycles=attention_cycles,
                             clock_mhz=hw.clock_mhz,
                             weight_bytes=weight_bytes)


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
