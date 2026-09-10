"""Perplexity on WikiText-2 for both architectures, host only.

    python -m kernel.experiments.ppl_paths --tokens 256

A rebuilds the dense key and then dots it. B never rebuilds it. They read the
same stored codes and the same quantised norm, so if the architecture choice is
free on accuracy these two numbers agree -- and the point of running it is that
"should agree" and "does agree" are different claims.

The baseline row is the unmodified model. The dense row is the kernel with
quantisation off, which exercises projection, the cast, RoPE, the rotation and
the fold with nothing compressed -- any gap there is the kernel's own
fixed-point error, and everything below it is the compression's.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.experiments.fpga_infer import MODEL                       # noqa: E402
from kernel.experiments.fpga_ppl import CHARS_PER_TOKEN, wikitext     # noqa: E402

MODES = [
    ("dense", "A   rebuild the dense key, then dot"),
    ("fused", "B   no rebuild, table form"),
]


def nll(model, ids):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        out = model(input_ids=ids.unsqueeze(0))
    logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
    return [-float(logp[i, int(ids[i + 1])]) for i in range(ids.shape[0] - 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key-bits", type=int, default=4)
    ap.add_argument("--value-bits", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None,
                    help="directory to write per-token NLL arrays to, as "
                         "<mode>_k<K>v<V>.npy. Without this the per-token "
                         "detail is discarded when the process exits, and only "
                         "the aggregate survives -- which is enough to report a "
                         "perplexity but not enough to pair two runs against "
                         "each other afterwards.")
    ap.add_argument("--modes", default="dense,fused",
                    help="which architectures to run: dense (A), fused (B), or both")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kernel import QuantConfig
    from kernel.adapters.torch_llama import graft, ungraft

    print(f"loading {a.model} on {a.device} (fp32)")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model.eval().to(a.device)

    text = wikitext(max(8000, a.tokens * CHARS_PER_TOKEN))
    ids = tok(text, return_tensors="pt")["input_ids"][0][:a.tokens].to(a.device)
    n_layers = model.config.num_hidden_layers
    print(f"{ids.shape[0]} tokens of WikiText-2, {n_layers} layers, "
          f"{a.key_bits}b keys / {a.value_bits}b values\n")

    base_nll = nll(model, ids)
    base = math.exp(sum(base_nll) / len(base_nll))
    per_token = {}
    print(f"    {'path':<36}{'perplexity':>12}{'vs baseline':>14}{'time':>8}")
    print(f"    {'baseline, unmodified model':<36}{base:>12.4f}{'—':>14}{'':>8}")

    quant = QuantConfig(key_bits=a.key_bits, value_bits=a.value_bits)
    wanted = [m.strip() for m in a.modes.split(",") if m.strip()]
    for m in wanted:
        if m not in dict(MODES):
            raise SystemExit(f"unknown mode {m!r}; expected dense and/or fused")
    for mode, label in [(m, l) for m, l in MODES if m in wanted]:
        t0 = time.time()
        original = graft(model, layers=list(range(n_layers)),
                         capacity=int(ids.shape[0]) + 1, quant=quant)
        kernels = [model.model.layers[L].self_attn.kernel for L in range(n_layers)]
        for k in kernels:
            k.attn.score_mode = mode
            k.attn.baseline_calls = 0
            # `attend_causal_batch` inlines its own scoring, so the batched
            # prefill path never reaches `scores()` and would run the fused
            # datapath whatever mode is selected. The token loop is the path
            # that honours it.
            #
            # B does not need it. The batched path is bit-identical to the loop
            # for the fused datapath -- that is pinned by
            # test_attention.py::check_causal_batch_is_bit_identical -- so when
            # only B is asked for, the loop buys nothing and costs an order of
            # magnitude. A still gets the loop, because for A the two paths are
            # not the same arithmetic.
            k.force_token_loop = (mode == "dense")
        try:
            per_token[mode] = np.array(nll(model, ids), dtype=np.float64)
            ppl = math.exp(per_token[mode].mean())
        finally:
            ungraft(model, original)
        if a.save:
            d = pathlib.Path(a.save); d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{mode}_k{a.key_bits}v{a.value_bits}.npy", per_token[mode])
        used = sum(k.attn.baseline_calls for k in kernels)
        if mode != "fused" and used == 0:
            raise SystemExit(
                f"mode {mode!r} was selected but never executed -- the scores "
                f"came from the fused path. Refusing to report it.")
        print(f"    {label:<36}{ppl:>12.4f}{ppl / base:>13.4f}x{time.time()-t0:>7.0f}s")

    if len(per_token) == 2:
        paired_report(per_token["dense"], per_token["fused"])
    else:
        print("\n  one architecture only; no paired comparison to report.")
    return 0


def paired_report(nll_a, nll_b):
    """A against B on the same tokens, so token difficulty cancels.

    Comparing two aggregate perplexities cannot separate a real effect from
    where two noisy averages happened to land: the token-to-token spread in
    NLL is far larger than anything the architecture does. Differencing per
    token removes that spread, because a hard token is hard under both.

    Reports a bound either way. If the effect is real the interval excludes
    zero; if it is not, the interval says how large it could still be, which
    is what "the choice is free on accuracy" actually needs.
    """
    d = nll_a - nll_b                       # positive => B is better
    n = d.size
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    lo, hi = mean - 1.96 * se, mean + 1.96 * se

    nz = d[d != 0.0]
    pos = int((nz > 0).sum())
    z = (pos - nz.size / 2) / math.sqrt(nz.size / 4) if nz.size else 0.0

    print(f"\n  paired over {n:,} tokens, d = NLL(A) - NLL(B), positive favours B")
    print(f"    mean d          {mean:+.6f}  (se {se:.6f})")
    print(f"    t               {mean / se if se else 0:+.2f}")
    print(f"    95% interval    [{lo:+.6f}, {hi:+.6f}]"
          f"{'  excludes zero' if lo * hi > 0 else '  includes zero'}")
    print(f"    B better on     {pos:,} of {nz.size:,} tokens that differ"
          f"   (sign-test z {z:+.1f})")
    print(f"    perplexity ratio B/A in "
          f"[{math.exp(-hi):.5f}, {math.exp(-lo):.5f}]")
    print("\n  Both read the same codes and the same quantised norm, so any")
    print("  difference here is the arithmetic and not the compression.")


if __name__ == "__main__":
    raise SystemExit(main())
