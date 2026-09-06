"""Compare the board's stored cache rows against the host's, byte for byte.

    ./scripts/attn diff

Why
---
Live inference agrees on token 1 and differs on every token after it, and the
mismatch is identical over JTAG and over Ethernet, with the cache zeroed. So
the transport is not it and the difference is deterministic.

Token 1 is the clue. At T = 1 the softmax runs over a single element, so
`p / l = 1` and the output is the decoded value alone -- the stored KEY cannot
affect it. A key row that is wrong is invisible at T = 1 and appears the moment
a second row exists, which is exactly the observed pattern.

That is a hypothesis about bytes in DDR, so this reads them. It steps the board
over JTAG (which can also read memory), then pulls the cache region back and
reassembles it with `DdrLayout.row_at` -- the same function that built every
image this project has ever loaded -- and diffs it against `kernel.cache.buf`.

The answer is one of three, and they point at different code:
  * the rows match  -> the caches agree and the divergence is in the scan
  * the key bytes differ -> the encoder or the write path, key plane
  * the value or norm bytes differ -> the same, but a different plane
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.experiments.fpga_infer import CACHE_BASE, MODEL          # noqa: E402
from kernel.host.attn_jtag import JtagClient                         # noqa: E402
from kernel.host.fpga_layers import FpgaLayers                       # noqa: E402
from kernel.hw.ddr_layout import DdrLayout                           # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt",
                    default="The history of computing hardware spans centuries")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval()

    ids = tok(a.prompt, return_tensors="pt")["input_ids"][0].tolist()[:a.tokens]
    capacity = len(ids) + 1
    print(f"{len(ids)} tokens, capacity {capacity}, layer {a.layer}")

    client = JtagClient()
    fpga = FpgaLayers(model, client, [a.layer], capacity,
                      cache_base=CACHE_BASE, verify=True)
    kern = fpga.mods[a.layer].kernel
    m = kern.cfg.model
    lay = DdrLayout.for_quant(kern.cfg.quant, m.head_dim, m.num_kv_heads,
                              capacity)

    past = None
    with torch.no_grad():
        for i, t in enumerate(ids):
            out = model(input_ids=torch.tensor([[t]]), past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            print(f"  token {i + 1}/{len(ids)} done")

    # -- the board's bytes --------------------------------------------------
    n = kern.cache.length
    base = fpga.bases[a.layer]
    print(f"reading {fpga.layer_stride} B of cache from {base:#x}")
    img = np.frombuffer(client._rd_bin(base, fpga.layer_stride), dtype=np.uint8)

    print()
    print(f"{'head':>5} {'token':>6} {'result':>10}  detail")
    bad_any = False
    for h in range(m.num_kv_heads):
        for t in range(n):
            got = lay.row_at(img, h, t)
            want = kern.cache.buf[t, h]
            if np.array_equal(got, want):
                if h == 0:
                    print(f"{h:5d} {t:6d} {'match':>10}")
                continue
            bad_any = True
            # Which field? `pack` lays a row out as key codes, then value
            # codes, then the two norms -- so a differing byte range names the
            # plane and therefore the part of the pipeline that wrote it.
            kb = m.head_dim * kern.cfg.quant.key_bits // 8
            vb = m.head_dim * kern.cfg.quant.value_bits // 8
            fields = (("key", 0, kb), ("value", kb, kb + vb),
                      ("norms", kb + vb, kb + vb + 4))
            which = [nm for nm, lo, hi in fields
                     if not np.array_equal(got[lo:hi], want[lo:hi])]
            nd = int(np.count_nonzero(got != want))
            print(f"{h:5d} {t:6d} {'DIFFER':>10}  {nd}/{got.size} B in "
                  f"{'+'.join(which)}")
            if h == 0 and t < 2:
                print(f"        board {got[:16].tolist()}")
                print(f"        host  {want[:16].tolist()}")

    print()
    if not bad_any:
        print("VERDICT: every stored row matches the host byte for byte.\n"
              "  The caches agree, so the divergence is in the SCAN, not in\n"
              "  what was written. Look at how the block reads them back.")
    else:
        print("VERDICT: the board's stored rows differ from the host's.\n"
              "  The field named above is the one whose encode-or-write path\n"
              "  is wrong. The scan is reading exactly what was put there.")
    client.quit()
    return 0 if not bad_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
