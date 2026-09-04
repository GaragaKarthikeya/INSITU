"""Golden vectors for the PL: the wire stream in, the expected answer out.

    python -m kernel.hw.vectors --ctx 64 --out tb/vectors

Every file this writes comes off the UNMODIFIED kernel. Nothing here computes a
value: it attaches `AttentionKernel.on_stage` -- an observer with the same
`is not None` guard the trace uses -- collects the arrays the datapath already
produced, and formats them. If a vector disagrees with the RTL, the numpy model
is right by construction, because it is the same code that scores
`cosine 1.000000` against `LlamaAttention`.

THE WIRE FORMAT, AND WHY IT NEEDS NO PADDING
--------------------------------------------
Q8.16 in 24-bit lanes on a 512-bit AXI4-Stream. `d = 64`, so one vector is
64 x 24 = 1,536 bits = **exactly 3 beats**. No `TKEEP` decoding, no DRE, no
partial beat -- but a lane still STRADDLES a beat boundary (24 does not divide
512), so `attn_ingress.sv` is a shift register and a counter, and this module
is the thing that says which bit goes where. Lanes are packed LSB-first into a
little-endian byte stream and beats are emitted MSB-first as hex, which is what
`$readmemh` wants.

GROUPED BY KV HEAD, AND THE ORDER IS CAUSALITY
----------------------------------------------
The stream is 8 groups of 6 vectors, `[k_h][v_h][q_4h..q_4h+3]`, so the block
can start on group 0 after 1,152 B instead of 9,216 B. **k and v come first
inside a group because the current token attends to itself**: its key and value
must be in the cache before any score is computed, exactly as
`cache.append_rotated` runs before attend in `kernel.py`. Getting this backwards
produces an off-by-one-token attention that still runs and still looks
plausible, which is the worst kind of wrong.

WHAT ELSE COMES OUT, AND FOR WHOM
---------------------------------
Each block in `plan.MD`'s table needs its own stimulus, so the per-stage
intermediates are emitted too: post-rotate for `rot_fwht.sv`, codes and norms
for `rot_norm.sv`/`rot_encode.sv` and for the `kv_store_ddr.sv` row layout, and
per-token scores for `score_lane.sv`.

The scores are taken from `CompressedAttention.scores` -- the same function the
kernel's own datapath calls, invoked here on the same rotated query and the
same cache view. It is not a reimplementation.

THE SOFTMAX GOLDEN IS `attend_online`
-------------------------------------
`forward` runs two-pass, and the two are NOT bit-identical -- the online form
truncates the accumulator once per rescale. The hardware is online, so
`out.online.hex` is what `attn_top.sv` must reproduce. `out.hex` is the
two-pass result `forward` returned, emitted beside it so the gap is visible
rather than assumed small. A testbench that checks against `out.hex` can never
pass; `plan.MD` says so and this file makes both available so the mistake is at
least an explicit one.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..config import FixedFormat, KernelConfig, ModelConfig, QuantConfig

BEAT_BITS = 512


# --------------------------------------------------------------------------
# 24-bit lane packing
# --------------------------------------------------------------------------

def pack_lanes(values, bits: int) -> bytes:
    """Signed integers -> LSB-first packed little-endian bytes.

    The last byte is zero-padded when `len(values) * bits` is not a whole
    number of bytes. For `d = 64` at 24 bits it always is, but the padding is
    written down rather than assumed away -- `KEY_BITS`/`VAL_BITS` packing uses
    the same rule and there it is not a whole number.
    """
    if bits <= 0:
        raise ValueError(f"bits={bits} must be positive")
    acc, n, out = 0, 0, bytearray()
    mask = (1 << bits) - 1
    for v in np.asarray(values).reshape(-1):
        acc |= (int(v) & mask) << n
        n += bits
        while n >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            n -= 8
    if n:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_lanes(buf: bytes, bits: int, count: int) -> np.ndarray:
    """The inverse, sign-extended. `pack_lanes` is only trustworthy with this."""
    acc = int.from_bytes(buf, "little")
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        v = (acc >> (i * bits)) & mask
        out[i] = (v ^ sign) - sign
    return out


def beats(buf: bytes, width_bits: int = BEAT_BITS) -> list[str]:
    """Packed bytes -> `$readmemh` lines, one beat each, MSB-first hex.

    Byte 0 of the stream is bits [7:0] of beat 0, so a beat prints its bytes
    REVERSED. That convention is the single most common way a testbench and a
    packer come to disagree while both look correct in isolation.
    """
    if width_bits % 8:
        raise ValueError(f"width_bits={width_bits} is not a whole number of bytes")
    w = width_bits // 8
    pad = (-len(buf)) % w
    b = buf + bytes(pad)
    return ["".join(f"{x:02x}" for x in reversed(b[i:i + w]))
            for i in range(0, len(b), w)]


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

@dataclass
class StageTap:
    """Whatever the kernel handed the observer, in order, per stage name."""

    seen: dict = field(default_factory=dict)

    def __call__(self, name: str, value) -> None:
        self.seen[name] = np.array(value, dtype=np.int64, copy=True)

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in self.seen:
            raise KeyError(f"stage {name!r} was never tapped; saw {sorted(self.seen)}")
        return self.seen[name]


@dataclass
class VectorSet:
    """One decode step's stimulus and expected results, ready to write."""

    model: ModelConfig
    quant: QuantConfig
    fmt: FixedFormat
    context: int                     # cached tokens BEFORE this step
    wire: dict = field(default_factory=dict)      # stage -> (tokens, heads, d)
    rotated: dict = field(default_factory=dict)
    quantizer: object | None = None               # the kernel's own KVQuantizer
    attn: object | None = None                    # the kernel's own CompressedAttention
    rows: np.ndarray | None = None                # (kv_heads, row_bytes) this token
    cache: np.ndarray | None = None               # (ctx+1, kv_heads, row_bytes) all
    scores: np.ndarray | None = None              # (heads, ctx+1) Q(acc_frac)
    out_two_pass: np.ndarray | None = None        # (tokens, hidden) Q(qk_frac)
    out_online: np.ndarray | None = None

    @property
    def lane_bits(self) -> int:
        return self.fmt.qk_width

    @property
    def beats_per_vector(self) -> int:
        total = self.model.head_dim * self.lane_bits
        if total % BEAT_BITS:
            raise ValueError(
                f"{self.model.head_dim} x {self.lane_bits} b = {total} b does not "
                f"divide the {BEAT_BITS} b bus; the stream would need TKEEP")
        return total // BEAT_BITS

    # -- the ingress stream ------------------------------------------------

    def ingress_groups(self) -> list[np.ndarray]:
        """8 groups of 6 vectors: `[k_h][v_h][q_4h..q_4h+3]`, token 0 only.

        k and v FIRST. See the module docstring -- this is causality, not
        layout preference.
        """
        m = self.model
        q, k, v = (self.wire["q"][0], self.wire["k"][0], self.wire["v"][0])
        out = []
        for h in range(m.num_kv_heads):
            heads = slice(h * m.kv_groups, (h + 1) * m.kv_groups)
            out.append(np.concatenate([k[h][None], v[h][None], q[heads]], axis=0))
        return out

    def ingress_bytes(self) -> bytes:
        return b"".join(pack_lanes(g, self.lane_bits) for g in self.ingress_groups())

    def ingress_beats(self) -> list[str]:
        return beats(self.ingress_bytes())


def collect(kernel, hidden, positions=None) -> VectorSet:
    """Run one step with the observer attached and gather everything.

    The kernel is left exactly as it was found, observer included, so a caller
    that reuses it for a second step is not silently still recording.
    """
    from ..ops.attention import CompressedAttention  # noqa: F401  (documented below)

    m = kernel.cfg.model
    ctx = kernel.cache.length
    tap = StageTap()
    prev, kernel.on_stage = kernel.on_stage, tap
    try:
        y, report = kernel.forward(hidden, positions)
    finally:
        kernel.on_stage = prev

    vs = VectorSet(model=m, quant=kernel.cfg.quant, fmt=kernel.cfg.fmt, context=ctx)
    for s in ("q", "k", "v"):
        vs.wire[s] = tap[f"wire.{s}"]
        vs.rotated[s] = tap[f"rot.{s}"]
    vs.out_two_pass = tap["out"]

    if kernel.compressed:
        # The encoder golden is the quantizer the cache is actually using, not
        # a second one built from the same config -- a codebook rebuilt here
        # would agree today and drift the first time `Codebook.build` changes.
        vs.quantizer = kernel.cache.quantizer
        vs.attn = kernel.attn

        # The row layout `kv_store_ddr.sv` must match byte for byte: the packed
        # bytes the cache actually stored, not a re-packing of them.
        vs.rows = kernel.cache.buf[ctx].copy()

        # The WHOLE cache, `ctx + 1` tokens including the one just appended.
        # `score_lane.sv` scans this; `kv_rows.hex` is only its last row, and a
        # bench fed one token cannot tell a working adder tree from a working
        # accumulator. Taken after `forward`, so the causal set the scores
        # below were computed over and the rows the bench streams are the same
        # `slice(0, ctx + 1)` -- not two slices that happen to agree today.
        vs.cache = kernel.cache.buf[:ctx + 1].copy()

        # Per-token scores and the online result, for `score_lane.sv` and
        # `softmax_online.sv`. Both come from the kernel's own attention
        # object, on the kernel's own cache -- the same calls `forward` makes.
        qr = vs.rotated["q"][0]
        sc, out_on = [], np.zeros((m.num_heads, m.head_dim), dtype=np.int64)
        for h in range(m.num_kv_heads):
            heads = slice(h * m.kv_groups, (h + 1) * m.kv_groups)
            kv = kernel.cache.view(h).select(slice(0, ctx + 1))
            s, _ = kernel.attn.scores(qr[heads], kv)
            sc.append(np.atleast_2d(s))
            y_on, _ = kernel.attn.attend_online(qr[heads], kv)
            out_on[heads] = y_on.reshape(m.kv_groups, m.head_dim)
        vs.scores = np.concatenate(sc, axis=0)
        from ..ops.project import merge_heads
        vs.out_online = merge_heads(out_on[None])
    return vs


# --------------------------------------------------------------------------
# the encoder: norms, codes, and the boundary artifact
# --------------------------------------------------------------------------

ENC_RANDOM = 512
ENC_TIES = 32
BOUND_BITS = 20          # widest boundary is 78,586 -- 18 b signed; 20 b on the wire
CENT_BITS  = 20          # widest centroid is 89,465 -- 18 b signed; same wire width


def _tie_vectors(quantizer, base: np.ndarray, plane) -> np.ndarray:
    """Vectors where a channel sits EXACTLY on a decision boundary.

    The `>` in `_encode_plane` sends a tie to the lower bin. That rule is one
    character of Python and one comparator bit in RTL, it disagrees with the
    obvious `>=`, and random stimulus will never hit it: a tie needs
    `x << 8 == norm * boundary` exactly, which has probability ~2**-32.

    Constructed rather than searched for. Set the last lane to the value that
    lands on a boundary, then recompute -- the change moves the norm, which
    moves the boundary, so it is iterated to a fixed point and only the vectors
    that actually converge are kept.
    """
    from ..numerics.fixed import INT

    shift = plane.threshold_frac - plane.qk_frac
    out = []
    for j, row in enumerate(base):
        v = row.copy()
        for _ in range(20):
            norm = int(quantizer._norm_wire(v[None])[0])
            # Only some boundaries land on a whole lane value at this norm.
            # Which one is picked is rotated across the stimulus so the ties
            # are spread over the bins rather than piled into one.
            usable = [int(norm) * int(b) for b in plane.codebook.boundaries
                      if (int(norm) * int(b)) % (1 << shift) == 0
                      and abs(int(norm) * int(b)) >> shift < (1 << 23)]
            if not usable:
                break
            cand = usable[j % len(usable)] >> shift
            if int(v[-1]) == cand:
                out.append(v.copy())       # fixed point: the tie is real
                break
            v[-1] = cand
    if len(out) < 8:
        raise AssertionError(f"only {len(out)} boundary-tie vectors converged")
    return np.array(out, dtype=INT)


def encoder_stimulus(vs: "VectorSet") -> np.ndarray:
    """Every vector the encoder benches run: real, random, and on-boundary."""
    real = np.concatenate([vs.rotated["k"][0], vs.rotated["v"][0]], axis=0)
    rng = np.random.default_rng(0xE17C0DE)
    # Bounded the way `rot_fwht`'s output is: this is the rotated domain, and a
    # vector the seam cannot carry is not a case the encoder has to encode.
    rand = rng.integers(-(1 << 21), 1 << 21,
                        size=(ENC_RANDOM, vs.model.head_dim), dtype=np.int64)
    ties = _tie_vectors(vs.quantizer, rand[:ENC_TIES], vs.quantizer.key)
    return np.concatenate([real, rand, ties], axis=0)


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

def _write(path: str, lines) -> str:
    with open(path, "w") as f:
        for ln in lines:
            f.write(f"{ln}\n")
    return path


def emit(vs: VectorSet, out_dir: str) -> list[str]:
    """Write every file. Returns the paths, in the order they were written."""
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)
    written = []

    written.append(_write(p("ingress.hex"), vs.ingress_beats()))
    for i, g in enumerate(vs.ingress_groups()):
        written.append(_write(p(f"ingress_group{i}.hex"),
                              beats(pack_lanes(g, vs.lane_bits))))

    # Per-stage intermediates, one vector per line at full lane width, so a
    # block-level TB can read a stage without decoding the 512-bit stream.
    w = (vs.lane_bits + 3) // 4
    for stage, book in (("wire", vs.wire), ("rot", vs.rotated)):
        for name, arr in book.items():
            written.append(_write(
                p(f"{stage}_{name}.hex"),
                ["".join(f"{int(x) & ((1 << vs.lane_bits) - 1):0{w}x}"
                         for x in reversed(vec))
                 for vec in arr.reshape(-1, vs.model.head_dim)]))

    # The randomized diagonal is a format artifact.  Hardware receives the
    # actual signs generated by Rotation, not a second implementation of
    # NumPy's PCG64 stream.  Bit i corresponds to lane i; one means negate.
    from ..ops.rotate import Rotation
    rot = Rotation.for_quant(vs.quant, vs.model.head_dim)
    if rot.rounds != 1 or rot.shift != 3:
        raise ValueError("rot_fwht vectors require one exact-shift d=64 round")
    sign_word = sum((int(s < 0) << i) for i, s in enumerate(rot.signs[0]))
    written.append(_write(p("rot_signs.hex"), [f"{sign_word:016x}"]))

    # Step 5's random regression is deliberately bounded so every normalized
    # output fits the 24-bit seam.  The golden is still Rotation.apply itself;
    # this only avoids testing a width the PL interface does not claim to hold.
    rng = np.random.default_rng(0xF17E_D64)
    r_in = rng.integers(-(1 << 19), 1 << 19,
                        size=(1024, vs.model.head_dim), dtype=np.int64)
    r_out = rot.apply(r_in)
    if np.any(r_out < -(1 << 23)) or np.any(r_out >= (1 << 23)):
        raise AssertionError("random FWHT golden escaped the 24-bit seam")
    written.append(_write(p("rot_random_in.hex"), [
        "".join(f"{int(x) & 0xFFFFFF:06x}" for x in reversed(row)) for row in r_in]))
    written.append(_write(p("rot_random_out.hex"), [
        "".join(f"{int(x) & 0xFFFFFF:06x}" for x in reversed(row)) for row in r_out]))

    # -- the encoder ---------------------------------------------------
    #
    # The boundaries are a format artifact exactly as the sign diagonal is:
    # hardware receives the integers `Codebook.build` produced, never a second
    # implementation of the Lloyd-Max solve.  Norms are plane-independent
    # (`_norm_wire` uses one `norm_q`), so one file serves both planes and the
    # same stimulus exercises KEY_BITS=4 and VAL_BITS=2 side by side.
    if vs.quantizer is not None:
        qz = vs.quantizer
        bw = (BOUND_BITS + 3) // 4
        for plane in (qz.key, qz.value):
            written.append(_write(
                p(f"enc_bounds_{plane.name}.hex"),
                [f"{int(b) & ((1 << BOUND_BITS) - 1):0{bw}x}"
                 for b in plane.codebook.boundaries]))

        x = encoder_stimulus(vs)
        norms = qz._norm_wire(x)
        written.append(_write(p("enc_in.hex"), [
            "".join(f"{int(v) & ((1 << vs.lane_bits) - 1):0{w}x}" for v in reversed(vec))
            for vec in x]))
        written.append(_write(p("enc_norm.hex"),
                              [f"{int(n) & 0xFFFF:04x}" for n in norms]))
        for plane in (qz.key, qz.value):
            idx = qz._encode_plane(x, plane, norms)
            written.append(_write(
                p(f"enc_idx_{plane.name}.hex"),
                ["".join(f"{int(c):02x}" for c in reversed(row)) for row in idx]))
        n_enc = int(x.shape[0])
    else:
        n_enc = 0

    if vs.rows is not None:
        written.append(_write(p("kv_rows.hex"),
                              ["".join(f"{b:02x}" for b in reversed(row))
                               for row in vs.rows]))

    # -- the score lane ------------------------------------------------
    #
    # One file per KV head, `ctx + 1` rows of `row_bytes`, in causal order --
    # exactly the byte stream `kv_store_ddr.sv` will hand `score_lane.sv`, and
    # exactly the bytes `pack` wrote. The lane splits the row itself rather
    # than being handed pre-split fields, so a bench that passes has agreed
    # with `_pack_codes`'s layout and not merely with a second decoder.
    if vs.cache is not None:
        for h in range(vs.model.num_kv_heads):
            written.append(_write(
                p(f"cache_rows_h{h}.hex"),
                ["".join(f"{b:02x}" for b in reversed(row))
                 for row in vs.cache[:, h]]))

    if vs.quantizer is not None:
        qz = vs.quantizer
        cw = (CENT_BITS + 3) // 4
        for plane in (qz.key, qz.value):
            written.append(_write(
                p(f"cb_centroids_{plane.name}.hex"),
                [f"{int(c) & ((1 << CENT_BITS) - 1):0{cw}x}"
                 for c in plane.codebook.centroids]))

        # The product tables `qtab_build.sv` must hold, one line per QUERY
        # head. Entry (ch, i) is `q_rot[ch] * centroid[i]` -- the multiply the
        # scan is amortising away -- at bit offset `(ch * 2**BITS + i) * PROD_W`,
        # LSB-first, the same convention every other packed file here uses.
        #
        # Emitted as the whole table rather than as the entries some token's
        # codes happen to select, so `tb_qtab_build` can sweep all 2**BITS
        # codes across all D channels and leave no untested entry.
        prod_w = vs.lane_bits + CENT_BITS - 1
        cents = qz.key.codebook.centroids.astype(np.int64)
        nb = int(cents.shape[0])
        mask = (1 << prod_w) - 1
        lines = []
        for q in vs.rotated["q"][0]:
            acc = 0
            for ch, x in enumerate(q):
                for i, c in enumerate(cents):
                    v = int(x) * int(c)
                    if not -(1 << (prod_w - 1)) <= v < (1 << (prod_w - 1)):
                        raise AssertionError(
                            f"q*centroid = {v} does not fit {prod_w} b; the "
                            f"product table width is wrong, not the vector")
                    acc |= (v & mask) << ((ch * nb + i) * prod_w)
            lines.append(f"{acc:0{(vs.model.head_dim * nb * prod_w + 3) // 4}x}")
        written.append(_write(p("qtab_table.hex"), lines))

        # -- rows a real cache does not contain --------------------------
        #
        # `_unpack_word` SIGN-EXTENDS the norm, so the model multiplies by a
        # signed 16-bit value. Every norm `isqrt` produces here is far below
        # 2**15, which means a lane that treated the norm as unsigned agrees
        # with `scores` on all 2,080 real rows and is still wrong. These rows
        # put the norm at 0, 1, 2**15-1, 2**15 and 2**16-1 -- the last two
        # negative once sign-extended -- and give the rule a case that can
        # actually fail. The codes are random because the norm is the point;
        # the golden is `CompressedAttention.scores` either way.
        from ..ops.quantize import CompressedKV

        rng = np.random.default_rng(0x5C0E)
        edge_norms = [0, 1, (1 << 15) - 1, 1 << 15, (1 << 16) - 1]
        d, nk, nv = vs.model.head_dim, 1 << vs.quant.key_bits, 1 << vs.quant.value_bits
        n_edge = len(edge_norms)
        kv = CompressedKV(
            k_idx=rng.integers(0, nk, size=(n_edge, d), dtype=np.uint8),
            k_norm=np.array([(n ^ (1 << 15)) - (1 << 15) for n in edge_norms],
                            dtype=np.int64),
            v_idx=rng.integers(0, nv, size=(n_edge, d), dtype=np.uint8),
            v_norm=np.array([(n ^ (1 << 15)) - (1 << 15) for n in edge_norms],
                            dtype=np.int64),
        )
        rows = qz.pack(kv)
        back = qz.unpack(rows)
        if not np.array_equal(back.k_norm, kv.k_norm):
            raise AssertionError("the constructed norms do not survive pack/unpack")
        edge, _ = vs.attn.scores(vs.rotated["q"][0][0], kv)
        written.append(_write(p("score_edge_rows.hex"),
                              ["".join(f"{b:02x}" for b in reversed(row))
                               for row in rows]))
        written.append(_write(p("score_edge_gold.hex"),
                              [f"{int(x) & 0xFFFFFFFFFFFF:012x}"
                               for x in np.ravel(edge)]))
    if vs.scores is not None:
        written.append(_write(p("scores.hex"),
                              ["".join(f"{int(x) & 0xFFFFFFFFFFFF:012x}"
                                       for x in reversed(row))
                               for row in vs.scores]))
    for name, arr in (("out.hex", vs.out_two_pass), ("out.online.hex", vs.out_online)):
        if arr is not None:
            written.append(_write(
                p(name), ["".join(f"{int(x) & 0xFFFFFF:06x}" for x in reversed(row))
                          for row in np.atleast_2d(arr)]))

    m = vs.model
    written.append(_write(p("manifest.json"), [json.dumps({
        "head_dim": m.head_dim, "num_heads": m.num_heads,
        "num_kv_heads": m.num_kv_heads, "kv_groups": m.kv_groups,
        "lane_bits": vs.lane_bits, "qk_frac": vs.fmt.qk_frac,
        "beat_bits": BEAT_BITS, "beats_per_vector": vs.beats_per_vector,
        "key_bits": vs.quant.key_bits, "value_bits": vs.quant.value_bits,
        "norm_bits": vs.quant.norm_bits, "norm_frac": vs.quant.norm_frac,
        "row_bytes": int(vs.rows.shape[-1]) if vs.rows is not None else 0,
        "context": vs.context, "rot_seed": vs.quant.seed,
        "rot_rounds": vs.quant.rot_rounds,
        "rot_sign_negative_is_one": True,
        "rot_random_vectors": int(r_in.shape[0]),
        "bound_bits": BOUND_BITS,
        "cent_bits": CENT_BITS,
        "prod_bits": vs.lane_bits + CENT_BITS - 1,
        "cache_tokens": int(vs.cache.shape[0]) if vs.cache is not None else 0,
        "score_edge_rows": 5,
        "enc_vectors": n_enc,
        "enc_real": 2 * m.num_kv_heads,
        "enc_random": ENC_RANDOM,
        "softmax_golden": "out.online.hex",
    }, indent=2)]))
    return written


# --------------------------------------------------------------------------
# the deterministic build
# --------------------------------------------------------------------------

def build(ctx: int = 64, seed: int = 0, key_bits: int = 4, value_bits: int = 2):
    """The reference configuration, from a seed and nothing else.

    Llama 3.2 1B's attention shape: 2048 hidden, 32 query heads, 8 KV heads,
    d = 64, no QK-norm. Weights and activations are pseudo-random from `seed`,
    which is what makes the vectors regenerate identically -- a real checkpoint
    would not be reproducible from a number.
    """
    from ..kernel import AttentionKernel

    m = ModelConfig(hidden_size=2048, num_heads=32, num_kv_heads=8, head_dim=64,
                    max_position=max(4096, ctx * 2 + 2))
    cfg = KernelConfig(model=m, quant=QuantConfig(key_bits=key_bits,
                                                  value_bits=value_bits, seed=seed))
    k = AttentionKernel(cfg, capacity=ctx + 1)
    rng = np.random.default_rng(seed)
    scale = 0.02
    if ctx:
        k.forward(rng.standard_normal((ctx, m.hidden_size)).astype(np.float32) * scale)
    step = rng.standard_normal((1, m.hidden_size)).astype(np.float32) * scale
    return collect(k, step), k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", type=int, default=64, help="cached tokens before the step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-bits", type=int, default=4)
    ap.add_argument("--value-bits", type=int, default=2)
    ap.add_argument("--out", default="tb/vectors")
    a = ap.parse_args(argv)

    vs, _ = build(ctx=a.ctx, seed=a.seed, key_bits=a.key_bits, value_bits=a.value_bits)
    paths = emit(vs, a.out)
    print(f"context {vs.context} -> {len(paths)} files in {a.out}")
    print(f"  ingress {len(vs.ingress_beats())} beats "
          f"({vs.beats_per_vector} per vector, {vs.lane_bits} b lanes)")
    print(f"  cache row {vs.rows.shape[-1]} B x {vs.model.num_kv_heads} heads")
    print(f"  golden softmax: out.online.hex (NOT out.hex -- see the docstring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
