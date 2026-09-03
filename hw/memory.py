"""The off-die port: weights and the KV cache behind one interface.

TWO KNOBS, SEPARATE ON PURPOSE
------------------------------
    latency   cycles from a burst being accepted to its first beat.
    beat_gap  idle cycles between beats once streaming has started.

They have different fixes and must not be collapsed into one "bandwidth"
number. Latency is hidden by prefetching deeper; a beat gap is not hidden by
anything -- it is simply less bandwidth. A model with one knob cannot tell you
which of those two you need.

THE PADDING IS REAL AND IS COUNTED
----------------------------------
A compressed token is 44 B on the wire but occupies a whole number of beats, so
at a 256-bit port it costs 64 B. Reporting the nominal 44 would understate DRAM
traffic by 45%. The counter-intuitive consequence -- a NARROWER port is more
byte-efficient here -- only shows up because the padding is charged rather than
assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import MemoryConfig
from ..trace import Op, Trace


@dataclass(frozen=True)
class MemoryReport:
    cycles: int
    bytes_read: int
    bytes_written: int
    beats: int
    bursts: int
    padding_bytes: int
    latency_cycles: int

    @property
    def efficiency(self) -> float:
        """Useful bytes over bytes the port actually moved."""
        moved = self.bytes_read + self.bytes_written + self.padding_bytes
        return (self.bytes_read + self.bytes_written) / moved if moved else 1.0

    def __str__(self) -> str:
        return (f"{self.cycles:,} cyc  {self.bytes_read:,} B read  "
                f"{self.bytes_written:,} B write  {self.bursts} bursts  "
                f"{self.padding_bytes:,} B beat padding  "
                f"efficiency {self.efficiency:.1%}")


class MemoryModel:
    """Replays a trace's MEM_READ / MEM_WRITE records into cycles and bytes.

    Bursts are charged individually: `latency` is per burst, not amortised over
    a step. A design that issues one burst per cached token pays the latency
    once per token, and that is the difference between a plausible model and a
    useful one.
    """

    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg

    def replay(self, trace: Trace) -> MemoryReport:
        c = self.cfg
        cycles = bursts = beats = 0
        read = write = padding = latency_total = 0

        for w in trace.records:
            if w.op is Op.MEM_READ:
                lat = c.read_latency
                read += w.n_bytes
            elif w.op is Op.MEM_WRITE:
                lat = c.write_latency
                write += w.n_bytes
            else:
                continue
            b = c.beats_for(w.n_bytes)
            padding += b * c.bytes_per_beat - w.n_bytes
            bursts += 1
            beats += b
            latency_total += lat
            cycles += lat + b * (1 + c.beat_gap)

        return MemoryReport(cycles=cycles, bytes_read=read, bytes_written=write,
                            beats=beats, bursts=bursts, padding_bytes=padding,
                            latency_cycles=latency_total)
