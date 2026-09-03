"""A systolic array, modelled at the PE.

DATAFLOW: OUTPUT-STATIONARY
---------------------------
Each PE owns one output element's partial sum. Weights stream in one column
per cycle; activations stream in one row per cycle. This is the choice that
decides everything below, so it is stated rather than implied.

The alternative -- weight-stationary -- holds a tile of weights in the PEs and
streams activations past them. It is the right choice when weights are reused,
which is to say when the batch is large. Decode has a batch of ONE.

WHY THAT MATTERS MORE THAN THE PE COUNT
---------------------------------------
A projection during decode is a matrix-VECTOR product. Every weight is used
exactly once. So the array's arithmetic intensity is 1 MAC per weight, and the
machine is bounded by how fast weights arrive, not by how many multipliers it
has. Doubling the array does nothing; doubling the weight bandwidth doubles
throughput.

This model reports both bounds separately (`compute_cycles`, `weight_cycles`)
and takes the max, so which one is binding is visible rather than inferred
from a single total. On a decode step it will be the weight bound by a wide
margin, and no array size fixes that.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ArrayConfig, MemoryConfig


@dataclass(frozen=True)
class ArrayReport:
    """Cycles for one projection, with the binding constraint left visible."""

    cycles: int
    compute_cycles: int
    weight_cycles: int
    fill_cycles: int
    tiles: int
    macs: int
    pe_count: int
    weight_bytes: int

    @property
    def bound_by(self) -> str:
        return "weights" if self.weight_cycles > self.compute_cycles else "compute"

    @property
    def utilization(self) -> float:
        """Fraction of the array's peak MAC throughput actually used."""
        peak = self.pe_count * self.cycles
        return self.macs / peak if peak else 0.0

    def __str__(self) -> str:
        return (f"{self.cycles:,} cyc  bound by {self.bound_by}  "
                f"util {self.utilization:6.2%}  "
                f"({self.tiles} tiles, {self.macs:,} MAC over {self.pe_count} PE)")


class SystolicArray:
    """Cycle model for one array. Reads a `Work` record; produces cycles.

    Holds no state between calls and never sees a value -- it is handed shapes.
    That is what guarantees the array's size cannot change a numeric result.
    """

    def __init__(self, cfg: ArrayConfig, mem: MemoryConfig) -> None:
        self.cfg = cfg
        self.mem = mem

    def project(self, m: int, k: int, n_tokens: int = 1) -> ArrayReport:
        """`(m, k) @ (k, n_tokens)` on a `rows x cols` array.

        Tiled `ceil(k/rows)` deep by `ceil(m/cols)` wide. Each tile:

            fill      rows + cols - 1, the systolic latency to prime and drain
            compute   n_tokens cycles, one activation column per cycle
            weights   rows*cols elements delivered over the port

        and the tile costs `fill + max(compute, weights)`. Tiles are not
        overlapped: overlapping them requires double-buffered weight registers,
        which is a real design decision with a real area cost, and assuming it
        for free is exactly the kind of flattery this model exists to avoid.
        """
        if m <= 0 or k <= 0 or n_tokens <= 0:
            raise ValueError(f"degenerate projection {m}x{k} over {n_tokens} tokens")
        c = self.cfg
        tiles_k = -(-k // c.rows)
        tiles_m = -(-m // c.cols)
        tiles = tiles_k * tiles_m

        fill = c.rows + c.cols - 1
        compute_per_tile = n_tokens * c.mul_latency
        weight_bytes_per_tile = c.rows * c.cols * self.mem.weight_dtype_bytes
        weight_per_tile = self.mem.beats_for(weight_bytes_per_tile) * (1 + self.mem.beat_gap)

        per_tile = fill + max(compute_per_tile, weight_per_tile)
        return ArrayReport(
            cycles=tiles * per_tile,
            compute_cycles=tiles * compute_per_tile,
            weight_cycles=tiles * weight_per_tile,
            fill_cycles=tiles * fill,
            tiles=tiles,
            macs=m * k * n_tokens,
            pe_count=c.pe_count,
            weight_bytes=tiles * weight_bytes_per_tile,
        )
