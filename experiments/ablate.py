"""Fast ablations on TinyLlama: check the design decisions before spending days.

WHY PAIRED, AND WHY SO FEW WINDOWS
----------------------------------
Every variant sees the SAME windows of the SAME text. The interesting quantity
is then the per-window difference in log-perplexity between two variants, and
window-to-window variance -- which is large, and which dominates an unpaired
comparison -- cancels exactly. That is what makes a 12-window run able to
resolve a 1% effect that a 12-window unpaired comparison could not.

Reported for every variant against the reference: the mean paired difference in
log-perplexity, its standard error over windows, and the perplexity ratio that
implies. A difference smaller than about twice its standard error is not
evidence of anything.

WHAT IS BEING CHECKED
---------------------
Things that were DESIGNED but never validated end to end:

  rounds     the rotation was changed from one round to two on a kurtosis
             measurement alone. Does it help the model?
  codebook   the Lloyd-Max Gaussian table is why no calibration file is needed.
             Does an equally-spaced table do just as well?
  seed       every number so far comes from one sign diagonal. Do others agree?
  softmax    the HARDWARE runs the online form; every number so far is the
             two-pass form. What does that cost?
  grid       the key/value asymmetry, on more than 400 words of text.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel import QuantConfig
from kernel.adapters.torch_llama import graft, ungraft
from kernel.experiments.runlog import RunLog

REF = "k4v2"          # the reference every paired comparison is against

def Q(**kw):
    return ("compressed", QuantConfig(key_bits=4, value_bits=2, **kw))

GROUPS = {
    "core": {
        "baseline":   (None, None),
        "dense":      ("dense", QuantConfig()),
        "k4v2":       Q(),                                  # the reference
        "rounds1":    Q(rot_rounds=1),
        "rounds3":    Q(rot_rounds=3),
        "uniform":    Q(codebook="uniform"),
        "seed1":      Q(seed=1),
        "seed2":      Q(seed=2),
        "seed3":      Q(seed=3),
        "seed4":      Q(seed=4),
    },
    "grid": {
        "baseline":   (None, None),
        "k4v2":       Q(),
        "k2v4":       ("compressed", QuantConfig(key_bits=2, value_bits=4)),
        "k3v4":       ("compressed", QuantConfig(key_bits=3, value_bits=4)),
        "k4v4":       ("compressed", QuantConfig(key_bits=4, value_bits=4)),
        "k6v4":       ("compressed", QuantConfig(key_bits=6, value_bits=4)),
        "k4v3":       ("compressed", QuantConfig(key_bits=4, value_bits=3)),
    },
    "softmax": {
        "baseline":   (None, None),
        "k4v2":       Q(),                                  # two-pass
        "k4v2online": Q(),                                  # same config, online form
    },
}


def load_windows(tok, context, n):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    return [ids[w * context:(w + 1) * context] for w in range(min(n, len(ids) // context))]


def set_softmax(model, mode):
    for layer in model.model.layers:
        a = layer.self_attn
        if hasattr(a, "kernel"):
            a.kernel.softmax = mode


def reset_all(model):
    for layer in model.model.layers:
        a = layer.self_attn
        if hasattr(a, "reset"):
            a.reset()


@torch.no_grad()
def window_logppl(model, w):
    ids = w.unsqueeze(0)
    return float(model(ids, labels=ids).loss)      # already mean NLL per token


def paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean and standard error of the per-window difference `a - b`."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="core", choices=list(GROUPS))
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--dir", default="kernel/experiments/results/ablate")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = Path(args.dir) / f"{args.group}-c{args.context}"
    log = RunLog(d)
    out = d / "results.json"
    state = json.loads(out.read_text()) if out.exists() else {}

    tok = AutoTokenizer.from_pretrained(args.model)
    wins = load_windows(tok, args.context, args.windows)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    layers = list(range(model.config.num_hidden_layers))
    n = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(
            lambda mod, inp, o, i=i: log.heartbeat(layer=i + 1, layers_total=n))

    variants = GROUPS[args.group]
    log.set(phase="running", group=args.group, context=args.context,
            windows=len(wins), total_variants=len(variants))
    log.event("start", group=args.group, context=args.context, windows=len(wins),
              variants=list(variants))
    print(f"{args.model}  ctx {args.context}  {len(wins)} windows  "
          f"group '{args.group}'  {len(variants)} variants\n", flush=True)

    for vi, (name, (mode, quant)) in enumerate(variants.items(), 1):
        if name in state and len(state[name]["logppl"]) == len(wins):
            print(f"{name:<11} cached"); continue
        original = None
        if mode is not None:
            original = graft(model, layers, quant=quant, capacity=args.context + 8,
                             compressed=(mode == "compressed"))
            if name.endswith("online"):
                set_softmax(model, "online")
        t0 = time.time()
        try:
            vals = []
            for wi, w in enumerate(wins):
                log.set(variant=name, config=name, variant_i=vi,
                        window=wi + 1, window_target=len(wins),
                        done_windows=(vi - 1) * len(wins) + wi,
                        total_windows=len(variants) * len(wins),
                        activity="forward")
                if original is not None:
                    reset_all(model)
                vals.append(window_logppl(model, w))
                log.heartbeat()
            bpt = (model.model.layers[0].self_attn.kernel.cache.bytes_per_token
                   if original is not None else None)
        finally:
            if original is not None:
                ungraft(model, original)
        state[name] = {"logppl": vals, "seconds": time.time() - t0,
                       "ppl": float(np.exp(np.mean(vals))), "windows": len(vals),
                       "bytes_per_token_per_layer": bpt}
        out.write_text(json.dumps(state, indent=1))
        elapsed = time.time() - log.started
        log.set(ppl={k: round(v.get("ppl", 0), 4) for k, v in state.items()},
                eta_h=round(elapsed / vi * (len(variants) - vi) / 3600, 2))
        log.event("variant", name=name, ppl=float(np.exp(np.mean(vals))),
                  seconds=round(time.time() - t0, 1))
        print(f"{name:<11} ppl {np.exp(np.mean(vals)):8.4f}  "
              f"{time.time() - t0:6.1f}s  ({vi}/{len(variants)})", flush=True)

    report(state, args.group, len(wins))
    log.set(phase="done"); log.event("done")


def report(state, group, n_windows):
    ref = state.get(REF)
    base = state.get("baseline")
    print(f"\n{'variant':<12} {'ppl':>9} {'vs base':>9} {'vs ' + REF:>10} "
          f"{'+- s.e.':>9} {'signif':>7} {'B/tok/L':>8}")
    for name, r in state.items():
        v = r["logppl"]
        ppl = float(np.exp(np.mean(v)))
        vb = f"{ppl / float(np.exp(np.mean(base['logppl']))):8.4f}x" if base else "        -"
        if ref and name not in (REF,):
            m, se = paired(v, ref["logppl"])
            ratio, sig = math.exp(m), ("yes" if abs(m) > 2 * se else "no")
            vr, ve = f"{ratio:9.4f}x", f"{se:9.4f}"
        elif name == REF:
            vr, ve, sig = "  (ref)", "        -", "-"
        else:
            vr, ve, sig = "        -", "        -", "-"
        print(f"{name:<12} {ppl:9.4f} {vb} {vr} {ve} {sig:>7} "
              f"{str(r.get('bytes_per_token_per_layer') or '-'):>8}")
    print(f"\npaired over {n_windows} windows; 'signif' = |difference| > 2 standard errors.")


if __name__ == "__main__":
    main()
