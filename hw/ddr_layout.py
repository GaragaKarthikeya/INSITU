"""Where a compressed KV cache lives in DDR, and why it is not laid out in rows.

THE FLAT ROW LAYOUT CANNOT FEED FOUR PORTS
------------------------------------------
`plan.MD` specified the cache as packed 52-byte rows at a flat stride, on the
reasoning that a sequential address pattern is what a prefetcher wants. It is --
but a row is 52 B and one 128-bit AXI-HP port carries 16 B per cycle, so the
engine needs FOUR ports running at once and each of them needs its own
sequential stream. A flat row layout gives you one stream, not four.

Splitting the token range into quarters -- port `p` reads tokens
`[pT/4, (p+1)T/4)` -- looks like it fixes that and does not. The scan is causal,
so the consumer wants token 0 first and every token of the first quarter comes
from port 0 alone. Port 0 delivers 16 B/cycle against the 52 B/cycle the score
lane eats, so it can only keep up if its whole quarter is already on chip; the
same holds for the other three, which is the entire cache resident in BRAM. That
is precisely the design this project exists to avoid.

SO THE CACHE IS TRANSPOSED: PLANES, NOT ROWS
--------------------------------------------
Split each row at its FIELD boundaries -- key codes, value codes, norms -- and
chop any field longer than a beat into beats. At 4b/2b that is four planes of
16, 16, 16 and 4 bytes:

    plane 0   row bytes  0..16   for every token      16 B/token
    plane 1   row bytes 16..32   for every token      16 B/token
    plane 2   row bytes 32..48   for every token      16 B/token
    plane 3   row bytes 48..52   for every token       4 B/token

Now every port reads ONE plane, perfectly sequentially, and the four together
deliver exactly one row per cycle: three ports at a beat per token and the
fourth at a beat per four tokens. 52 B/cycle out of 64 B/cycle of interface,
with no reassembly buffer beyond a burst and no port ever reading out of order.

The port count falls out rather than being chosen -- it is `ceil(row_bytes/16)`,
which is the number `plan.MD` already derived from the bandwidth side.

WHY FIELDS AND NOT JUST EVERY 16 BYTES
--------------------------------------
Chopping the row blindly every 16 bytes is simpler and is wrong for half the
`(key_bits, value_bits)` grid step 15 sweeps. A row is `8*kb + 8*vb + 4` bytes,
so `kb + vb = 3` gives 28 B and a trailing plane of 12 -- and 12 neither divides
a 16-byte beat nor is a multiple of one, so tokens would straddle beats and that
port would need a byte shift register.

Splitting at fields first cannot produce that. Every field is `8*kb`, `8*vb` or
4 bytes; the first two are 0 or 8 modulo 16 and the last is 4, so every plane is
16, 8 or 4 bytes wide and all three divide the beat exactly.
`check_every_quantisation_width_gives_beat_friendly_planes` sweeps all 64
combinations.

THE ROW FORMAT IS UNCHANGED
---------------------------
This is a layout, not a format. The bytes are the bytes `KVQuantizer.pack`
wrote, in the order it wrote them; they are grouped differently in memory.
`row_at` reassembles one and `tests/test_ddr_layout.py` asserts the round trip
is byte-exact against `pack` for every token and every head, which is the only
property anything downstream depends on.

PLANES ARE 4 KB ALIGNED, AND PACKED TIGHT WITHIN
------------------------------------------------
Each plane starts on a 4 KB boundary so a burst never straddles a DRAM page
boundary it did not have to. Within a plane the stride is the plane's own width
-- 4 B for the norms, not 16 -- because padding it to a beat would make that
port read 16 B/token and push the engine from 52 B/cycle to 64, which is the
interface ceiling exactly and leaves the DRAM controller no room at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BEAT_BYTES = 16          # one 128-bit AXI-HP port
PAGE = 4096              # plane alignment


@dataclass(frozen=True)
class Plane:
    """One port's stream: a byte range of the row, for every token."""

    index: int
    lo: int              # first row byte, inclusive
    hi: int              # last row byte, exclusive

    @property
    def width(self) -> int:
        return self.hi - self.lo

    @property
    def tokens_per_beat(self) -> float:
        return BEAT_BYTES / self.width


@dataclass(frozen=True)
class DdrLayout:
    """The address map. One instance is the contract between host and PL."""

    row_bytes: int
    n_kv_heads: int
    capacity: int
    base: int = 0
    # (lo, hi) byte ranges of the row's fields, in `pack`'s order. Defaults to
    # one field spanning the whole row, which is the blind-chunking behaviour
    # and is only correct when `row_bytes` is a multiple of the beat.
    fields: tuple = ()

    @classmethod
    def for_quant(cls, quant, head_dim, n_kv_heads, capacity, base: int = 0):
        """The layout for a `QuantConfig`, with `pack`'s field boundaries.

        Built from the same three numbers `KVQuantizer.pack` lays a row out
        from, so the two cannot drift: key codes, then value codes, then the
        two norms.
        """
        kb = head_dim * quant.key_bits // 8
        vb = head_dim * quant.value_bits // 8
        nb = 2 * (quant.norm_bits // 8)
        if head_dim * quant.key_bits % 8 or head_dim * quant.value_bits % 8:
            raise ValueError("a code plane is not a whole number of bytes")
        return cls(row_bytes=kb + vb + nb, n_kv_heads=n_kv_heads,
                   capacity=capacity, base=base,
                   fields=((0, kb), (kb, kb + vb), (kb + vb, kb + vb + nb)))

    def __post_init__(self) -> None:
        if self.row_bytes <= 0:
            raise ValueError(f"row_bytes={self.row_bytes} must be positive")
        if not self.fields:
            object.__setattr__(self, "fields", ((0, self.row_bytes),))
        if self.fields[0][0] != 0 or self.fields[-1][1] != self.row_bytes:
            raise ValueError(f"fields {self.fields} do not tile {self.row_bytes} B")
        for a, b in zip(self.fields, self.fields[1:]):
            if a[1] != b[0]:
                raise ValueError(f"fields {self.fields} leave a gap or overlap")
        # Every plane must be whole beats per token or whole tokens per beat,
        # or that port needs a byte shift register and the transpose has bought
        # nothing. Field-aligned splitting guarantees it; blind chunking does
        # not, which is why `fields` exists.
        for pl in self.planes:
            if pl.width > BEAT_BYTES or BEAT_BYTES % pl.width:
                raise ValueError(
                    f"plane {pl.index} is {pl.width} B/token, which neither "
                    f"divides nor is a multiple of the {BEAT_BYTES} B beat; "
                    f"split the row at its field boundaries"
                )

    @property
    def planes(self) -> tuple[Plane, ...]:
        """Field boundaries first, then beats. The port count is not a choice.

        `fields` is the row's own structure -- what `KVQuantizer.pack` wrote, in
        the order it wrote it -- so a plane is always part of exactly one field.
        """
        out, idx = [], 0
        for lo, hi in self.fields:
            a = lo
            while a < hi:
                b = min(a + BEAT_BYTES, hi)
                out.append(Plane(idx, a, b))
                idx += 1
                a = b
        return tuple(out)

    @property
    def n_ports(self) -> int:
        return len(self.planes)

    def plane_span(self, p: int) -> int:
        """Bytes one plane occupies for one head. The SAME for every plane.

        EVERY PLANE GETS THE WIDEST PLANE'S SPAN, AND THAT IS THE HARDWARE'S
        CONSTRAINT SPEAKING
        ------------------------------------------------------------------
        The obvious map gives each plane exactly the pages its own width needs
        -- at capacity 1,025 that is 20,480 B for the three code planes and
        8,192 B for the 4-byte norms. `attn_top.sv` cannot address it. Its
        address generator is one register, `plane_span`, and it walks the four
        planes as `cache_base + p * plane_span`, because per-plane bases were a
        49-bit multiply on the path that missed 250 MHz by 1.953 ns (step 12,
        fix 2). Four spans means four base registers the host has to load, or a
        multiplier back on that path.

        So the norms plane is padded out to the code planes' span. It costs
        address space and nothing else: the padding sits after the last token
        of the plane, the port reads 4 B/token sequentially from the base as
        before, and no read is ever issued to it. At capacity 32,769 that is
        16.9 MB against 13.8 MB, in a 2 GB DDR.

        Nothing about this shows up at capacity 65 -- there all four planes
        round to one page anyway, which is why every vector set and every bench
        in this project agreed with a layout the hardware could not have read.
        `check_every_plane_has_the_same_span` is what says so at other sizes.
        """
        raw = max(self.capacity * pl.width for pl in self.planes)
        return (raw + PAGE - 1) // PAGE * PAGE

    def plane_base(self, head: int, p: int) -> int:
        """Base address of head `head`'s plane `p`.

        Head-major so that one head's four planes are adjacent: a scan touches
        exactly four regions, and consecutive scans of the same layer walk the
        same four. Plane-major would spread one head's four streams across the
        whole cache and give the DRAM controller nothing to keep open.
        """
        addr = self.base
        for h in range(head):
            addr += sum(self.plane_span(i) for i in range(self.n_ports))
        for i in range(p):
            addr += self.plane_span(i)
        return addr

    def token_addr(self, head: int, p: int, token: int) -> int:
        return self.plane_base(head, p) + token * self.planes[p].width

    @property
    def total_bytes(self) -> int:
        return self.n_kv_heads * sum(self.plane_span(i) for i in range(self.n_ports))

    @property
    def bytes_per_cycle(self) -> int:
        """What the four ports must deliver to retire one row per cycle."""
        return self.row_bytes

    # -- the image ---------------------------------------------------------

    def image(self, buf: np.ndarray) -> np.ndarray:
        """`cache.buf[:n]` -> the DDR image, as bytes.

        `buf` is (tokens, kv_heads, row_bytes), which is exactly what
        `CompressedCache` holds. The transpose happens here and nowhere else.
        """
        buf = np.asarray(buf, dtype=np.uint8)
        n, heads, rb = buf.shape
        if rb != self.row_bytes:
            raise ValueError(f"row is {rb} B, layout is built for {self.row_bytes}")
        if n > self.capacity:
            raise ValueError(f"{n} tokens exceeds capacity {self.capacity}")
        if heads != self.n_kv_heads:
            raise ValueError(f"{heads} kv heads, layout has {self.n_kv_heads}")

        img = np.zeros(self.total_bytes, dtype=np.uint8)
        for h in range(heads):
            for pl in self.planes:
                start = self.plane_base(h, pl.index) - self.base
                chunk = buf[:n, h, pl.lo:pl.hi].reshape(-1)
                img[start:start + chunk.size] = chunk
        return img

    def row_at(self, img: np.ndarray, head: int, token: int) -> np.ndarray:
        """The inverse: reassemble one 52-byte row out of the four planes."""
        img = np.asarray(img, dtype=np.uint8)
        parts = []
        for pl in self.planes:
            a = self.token_addr(head, pl.index, token) - self.base
            parts.append(img[a:a + pl.width])
        return np.concatenate(parts)

    # -- what the RTL is parameterised by ----------------------------------

    def describe(self) -> dict:
        return {
            "row_bytes": self.row_bytes,
            "beat_bytes": BEAT_BYTES,
            "n_ports": self.n_ports,
            "plane_width": [pl.width for pl in self.planes],
            "plane_span": [self.plane_span(i) for i in range(self.n_ports)],
            "head_stride": sum(self.plane_span(i) for i in range(self.n_ports)),
            "capacity": self.capacity,
            "n_kv_heads": self.n_kv_heads,
            "base": self.base,
            "total_bytes": self.total_bytes,
            "bytes_per_cycle": self.bytes_per_cycle,
        }
