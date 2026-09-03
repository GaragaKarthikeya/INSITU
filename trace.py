"""The seam between what a value is and when it arrives.

Ops compute values and, if a trace is attached, append a `Work` record saying
what they did in hardware terms. They never compute a cycle count, never read a
`HardwareConfig`, and never branch on whether tracing is on beyond one `is not
None` check.

The hardware models in `kernel.hw` consume the trace afterwards and turn it
into cycles. Nothing flows back.

WHY THIS SHAPE
--------------
It makes an invariant mechanical rather than aspirational: **the hardware
configuration cannot change a numeric result**, because no op can see it. That
is what lets the same kernel be a plain attention module inside a real model
(no trace) and an accelerator model (trace attached) without two code paths
that must be kept in agreement by hand.

It also means a trace is a complete, inspectable account of a step. A
bytes/token figure is read off `MEM_READ` records, not declared in a document
and hoped to still be true.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class Op(Enum):
    """What kind of work. One entry per unit that costs something."""

    MATVEC = "matvec"        # a projection through a systolic array
    ROPE = "rope"
    ROTATE = "rotate"        # gather -> sign flip -> FWHT -> scale
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"
    SCORE = "score"          # one query against one cached key
    ACCUMULATE = "accumulate"
    SOFTMAX = "softmax"
    RESCALE = "rescale"      # the online softmax pulling the accumulator down
    MEM_READ = "mem_read"
    MEM_WRITE = "mem_write"


@dataclass(frozen=True)
class Work:
    """One unit of work. Shapes and byte counts only -- never a cycle count.

    `unit` names which physical resource did it (`"q_array"`, `"memory"`, ...)
    so a trace can be replayed against a hardware config that has more than one
    of a thing, without the op needing to know how many there are.
    """

    op: Op
    unit: str = ""
    m: int = 0              # output width  / element count
    k: int = 0              # reduction depth
    n: int = 1              # how many such operations
    n_bytes: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def macs(self) -> int:
        return self.m * self.k * self.n


class Trace:
    """An append-only log of `Work`.

    Cheap enough to leave on: appending a frozen dataclass per *stage* (not per
    element) costs a few microseconds against milliseconds of numpy.
    """

    __slots__ = ("records", "_labels")

    def __init__(self) -> None:
        self.records: list[Work] = []
        self._labels: list[str] = []

    # -- recording ---------------------------------------------------------

    def add(self, op: Op, unit: str = "", *, m: int = 0, k: int = 0,
            n: int = 1, n_bytes: int = 0, **meta) -> None:
        if self._labels:
            meta = {**meta, "phase": self._labels[-1]}
        self.records.append(Work(op, unit, m, k, n, n_bytes, meta))

    def phase(self, label: str) -> "_Phase":
        """`with trace.phase("decode"): ...` tags everything inside."""
        return _Phase(self, label)

    # -- reading -----------------------------------------------------------

    def of(self, op: Op) -> list[Work]:
        return [w for w in self.records if w.op is op]

    def bytes_read(self) -> int:
        return sum(w.n_bytes for w in self.of(Op.MEM_READ))

    def bytes_written(self) -> int:
        return sum(w.n_bytes for w in self.of(Op.MEM_WRITE))

    def macs(self) -> int:
        return sum(w.macs for w in self.records)

    def counts(self) -> Counter:
        return Counter(w.op.value for w in self.records)

    def filter(self, *, phase: str | None = None, unit: str | None = None) -> "Trace":
        out = Trace()
        out.records = [
            w for w in self.records
            if (phase is None or w.meta.get("phase") == phase)
            and (unit is None or w.unit == unit)
        ]
        return out

    def summary(self) -> str:
        c = self.counts()
        return (
            f"{len(self.records)} records  "
            f"{self.macs():,} MAC  "
            f"read {self.bytes_read():,} B  write {self.bytes_written():,} B\n"
            + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
        )

    def __len__(self) -> int:
        return len(self.records)


class _Phase:
    __slots__ = ("trace", "label")

    def __init__(self, trace: Trace, label: str) -> None:
        self.trace, self.label = trace, label

    def __enter__(self) -> Trace:
        self.trace._labels.append(self.label)
        return self.trace

    def __exit__(self, *exc) -> None:
        self.trace._labels.pop()
        return False


def record(trace: Trace | None, op: Op, unit: str = "", **kw) -> None:
    """Append if there is a trace. The single guard every op uses."""
    if trace is not None:
        trace.add(op, unit, **kw)
