"""The vectors the A53 runs on the board: many steps, at several contexts.

WHY THIS IS NOT `vectors.py --ctx N`
------------------------------------
`hw/vectors.py` emits ONE decode step, because that is what a testbench needs:
a bench that cannot reproduce a single step will not be helped by eight. Step
13's exit criterion is the other shape -- "multi-token, multi-context,
bit-exact" -- and the reason is a property no single step can test.

A decode step WRITES its own cache row to DDR through the block's own AXI write
master and then reads it back in the same scan, because the token attends to
itself. `tb_attn` checks that by blanking token 64 and watching it come back.
What neither the bench nor step 12 checks is whether that row is still there
for the NEXT step -- whether the write landed where the next scan's address
generator will look, in a plane the next scan will read all the way to. A run
of `steps` tokens from one loaded image checks exactly that: token `ctx0 + n`
scores against `n` rows that exist only because the hardware wrote them, and a
write master that were one plane or one token off would answer correctly for
one step and wrongly for every step after it.

So the image loaded here holds tokens `0 .. ctx0-1` and nothing else. Tokens
`ctx0 ..` are ZERO in every plane of every head, and the only way they become
the right bytes is the block's own write path.

WHY SEVERAL CONTEXTS, AND WHY THE LARGEST IS THE ONE THAT MEANS ANYTHING
------------------------------------------------------------------------
Step 13 measured it: a scan is 3.87 cycles per row at ctx 64, 1.83 at 256 and
1.24 at 1,024, because eight groups each pay a pipeline fill and a DDR round
trip that a short scan cannot amortise. A bandwidth divided by the ctx-64
number is a statement about that overhead, not about DDR. ctx 8,192 is here
because it is the first row of `plan.MD`'s own throughput table -- the regime
every headline number in this project is quoted at -- and at 1,026 rows a
group the fill is under 3%. The short cases stay because a bug that only
appears at length is far easier to find once the short case is known good.

The ctx-8,192 image is 4.1 MB and it is carried in the ELF, which makes the
JTAG download the slow part of a board session. That is a one-time cost and no
part of any number the run reports.

THE C FILE IS GENERATED, NOT WRITTEN
------------------------------------
Everything the board compares against comes out of the same numpy kernel the
rest of the project is checked against, through the same `collect` the
testbenches use. There is no second golden model in C: `sw/attn_main.c` holds
control flow and no arithmetic, so there is nothing in it that can be right
about attention and wrong about this.

Usage:
    python -m kernel.hw.board_vectors --out sw
"""

from __future__ import annotations

import argparse
import struct

import numpy as np

from .ddr_layout import DdrLayout
from .vectors import collect, pack_lanes

# (name, cached tokens before the first step, decode steps)
#
# 4 steps is the smallest number that distinguishes the three ways a write path
# can be wrong: right row (1 would pass), right plane (2), right stride (3+).
CASES = (("ctx64", 64, 4), ("ctx256", 256, 4), ("ctx1024", 1024, 4),
         ("ctx8192", 8192, 4))

IN_WORDS = 2304          # 8 groups x 6 vectors x 3 beats x 16 words
OUT_WORDS = 1536         # 2,048 lanes of 24 b


def _u32(buf: bytes) -> np.ndarray:
    """Packed bytes -> the 32-bit words the shim's buffers hold.

    Byte 0 of the stream is the low byte of word 0, at the lowest address.
    That is the feeder's convention (`attn_ctrl.sv` shifts `inbuf[0]` in first,
    low word first) and the packer's (`pack_lanes` is LSB-first little-endian),
    and they are the same convention -- which is the only reason this is a
    reinterpret and not a permutation.
    """
    pad = (-len(buf)) % 4
    return np.frombuffer(buf + bytes(pad), dtype="<u4")


def build_case(ctx0: int, steps: int, seed: int = 0,
               key_bits: int = 4, value_bits: int = 2) -> dict:
    """Run `ctx0` tokens to fill a cache, then `steps` decode steps on it."""
    from ..config import KernelConfig, ModelConfig, QuantConfig
    from ..kernel import AttentionKernel

    capacity = ctx0 + steps
    m = ModelConfig(hidden_size=2048, num_heads=32, num_kv_heads=8, head_dim=64,
                    max_position=max(4096, 2 * capacity + 2))
    cfg = KernelConfig(model=m, quant=QuantConfig(key_bits=key_bits,
                                                  value_bits=value_bits, seed=seed))
    k = AttentionKernel(cfg, capacity=capacity)
    rng = np.random.default_rng(seed)
    scale = 0.02
    if ctx0:
        k.forward(rng.standard_normal((ctx0, m.hidden_size)).astype(np.float32) * scale)

    lay = DdrLayout.for_quant(cfg.quant, m.head_dim, m.num_kv_heads, capacity)

    # The image as of BEFORE the first step: tokens 0..ctx0-1 only. `image`
    # zero-fills the rest of every plane, which is exactly the blanking
    # `tb_attn` does by hand -- the tokens the block must write for itself.
    image = lay.image(k.cache.buf[:ctx0])

    ingress, golden, n_tokens = [], [], []
    for _ in range(steps):
        step = rng.standard_normal((1, m.hidden_size)).astype(np.float32) * scale
        n_tokens.append(k.cache.length + 1)
        vs = collect(k, step)
        iw = _u32(vs.ingress_bytes())
        gw = _u32(pack_lanes(vs.out_online, vs.lane_bits))
        if iw.size != IN_WORDS:
            raise AssertionError(f"ingress is {iw.size} words, expected {IN_WORDS}")
        if gw.size != OUT_WORDS:
            raise AssertionError(f"golden is {gw.size} words, expected {OUT_WORDS}")
        ingress.append(iw)
        golden.append(gw)

    # The image the board loads must be the one the LAST step's scan reads --
    # minus the rows the hardware writes. Check the surviving rows here, where
    # a mismatch is a Python error and not a wrong answer on a board.
    for h in range(m.num_kv_heads):
        for t in range(ctx0):
            if not np.array_equal(lay.row_at(image, h, t), k.cache.buf[t, h]):
                raise AssertionError(f"image loses head {h} token {t}")
        for t in range(ctx0, capacity):
            if lay.row_at(image, h, t).any():
                raise AssertionError(
                    f"head {h} token {t} is in the image; the block must write it")

    d = lay.describe()
    return {
        "ctx0": ctx0, "steps": steps, "capacity": capacity,
        "head_stride": d["head_stride"], "plane_span": d["plane_span"][0],
        "n_kv_heads": d["n_kv_heads"], "row_bytes": d["row_bytes"],
        "image": image, "ingress": np.concatenate(ingress),
        "golden": np.concatenate(golden),
        "n_tokens": np.array(n_tokens, dtype="<u4"),
    }


# --------------------------------------------------------------------------
# emission
# --------------------------------------------------------------------------

def _c_bytes(name: str, arr: np.ndarray) -> str:
    b = np.asarray(arr, dtype=np.uint8).tobytes()
    rows = ["    " + "".join(f"{x:#04x}," for x in b[i:i + 16])
            for i in range(0, len(b), 16)]
    return (f"const unsigned char {name}[{len(b)}] "
            f"__attribute__((aligned(64))) = {{\n" + "\n".join(rows) + "\n};\n")


def _c_words(name: str, arr: np.ndarray) -> str:
    w = np.asarray(arr, dtype="<u4")
    rows = ["    " + "".join(f"{int(x):#010x}," for x in w[i:i + 8])
            for i in range(0, w.size, 8)]
    return (f"const unsigned int {name}[{w.size}] = {{\n"
            + "\n".join(rows) + "\n};\n")


HEADER = '''/* GENERATED by `python -m kernel.hw.board_vectors` -- do not edit.
 *
 * The board's goldens, from the same numpy kernel every other check in this
 * project is measured against, through the same `collect` the testbenches use.
 * `attn_main.c` holds control flow and no arithmetic, so there is nothing in it
 * that can be right about attention and wrong about these numbers.
 *
 * Each case is one loaded cache image and `steps` decode steps run back to
 * back on it. The image holds tokens 0..ctx0-1 ONLY: every later token is zero
 * in every plane, and the only way it becomes the right bytes is the block's
 * own AXI write master. Step n therefore scores against n rows that exist
 * because the hardware put them there.
 */
#ifndef ATTN_VECTORS_H
#define ATTN_VECTORS_H

#define ATTN_IN_WORDS   %d
#define ATTN_OUT_WORDS  %d

typedef struct {
    const char         *name;
    unsigned            ctx0;         /* cached tokens before the first step */
    unsigned            steps;
    unsigned            capacity;     /* ctx0 + steps */
    unsigned            head_stride;  /* bytes per KV head in the image */
    unsigned            plane_span;   /* bytes per plane; the same for all four */
    unsigned            n_kv_heads;
    unsigned            row_bytes;    /* what one token costs the four ports */
    unsigned            image_bytes;
    const unsigned char *image;
    const unsigned int  *n_tokens;    /* [steps] -- ctx+1 for each step */
    const unsigned int  *ingress;     /* [steps * ATTN_IN_WORDS] */
    const unsigned int  *golden;      /* [steps * ATTN_OUT_WORDS] */
} attn_case_t;

extern const attn_case_t attn_cases[];
extern const unsigned    attn_n_cases;

#endif
'''


def emit(cases: list[dict], out_dir: str) -> list[str]:
    import os

    written = []
    hdr = os.path.join(out_dir, "attn_vectors.h")
    with open(hdr, "w") as f:
        f.write(HEADER % (IN_WORDS, OUT_WORDS))
    written.append(hdr)

    src = os.path.join(out_dir, "attn_vectors.c")
    with open(src, "w") as f:
        f.write('/* GENERATED by `python -m kernel.hw.board_vectors` '
                '-- do not edit. */\n#include "attn_vectors.h"\n\n')
        for c in cases:
            tag = c.get("name", f"c{c['ctx0']}").replace("-", "_")
            f.write(_c_bytes(f"img_{tag}", c["image"]))
            f.write(_c_words(f"ntok_{tag}", c["n_tokens"]))
            f.write(_c_words(f"in_{tag}", c["ingress"]))
            f.write(_c_words(f"gold_{tag}", c["golden"]))
            f.write("\n")
        f.write("const attn_case_t attn_cases[] = {\n")
        for c in cases:
            name = c.get("name", f"ctx{c['ctx0']}")
            tag = name.replace("-", "_")
            f.write(
                f'    {{ "{name}", {c["ctx0"]}, {c["steps"]}, '
                f'{c["capacity"]}, {c["head_stride"]}, {c["plane_span"]}, '
                f'{c["n_kv_heads"]}, {c["row_bytes"]}, {c["image"].size}, '
                f'img_{tag}, ntok_{tag}, in_{tag}, gold_{tag} }},\n')
        f.write("};\nconst unsigned attn_n_cases = "
                f"{len(cases)};\n")
    written.append(src)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="sw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-bits", type=int, default=4)
    ap.add_argument("--value-bits", type=int, default=2)
    a = ap.parse_args(argv)

    cases = []
    total = 0
    for name, ctx0, steps in CASES:
        c = build_case(ctx0, steps, seed=a.seed, key_bits=a.key_bits,
                       value_bits=a.value_bits)
        cases.append(c)
        bytes_ = c["image"].size + 4 * (c["ingress"].size + c["golden"].size)
        total += bytes_
        print(f"  {name}: {steps} steps, capacity {c['capacity']}, "
              f"image {c['image'].size} B, span {c['plane_span']}, "
              f"stride {c['head_stride']} -> {bytes_/1024:.0f} KiB")
    paths = emit(cases, a.out)
    print(f"{len(cases)} cases, {total/1024:.0f} KiB of data -> "
          + ", ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
