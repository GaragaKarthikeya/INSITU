"""Fast ablations on TinyLlama: check the design decisions before spending days.

Why paired, and why so few windows
----------------------------------
Every variant sees the same windows of the same text. The interesting quantity
is then the per-window difference in log-perplexity between two variants, and
window-to-window variance -- which is large, and which dominates an unpaired
comparison -- cancels exactly. That is what makes a 12-window run able to
resolve a 1% effect that a 12-window unpaired comparison could not.

Reported for every variant against the reference: the mean paired difference in
log-perplexity, its standard error over windows, and the perplexity ratio that
implies. A difference smaller than about twice its standard error is not
evidence of anything.

What is being checked
---------------------
Things that were designed but never validated end to end:

  rounds     the rotation was changed from one round to two on a kurtosis
             measurement alone. Does it help the model?
  codebook   the Lloyd-Max Gaussian table is why no calibration file is needed.
             Does an equally-spaced table do just as well?
  seed       every number so far comes from one sign diagonal. Do others agree?
  softmax    the hardware runs the online form; every number so far is the
             two-pass form. What does that cost?
  grid       the key/value asymmetry, on more than 400 words of text.
  rotation   does the rotation do anything at all? Every other line in this
             file compares one rotation against another and takes rotating
             for granted. See `ROTATION` below.
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

REF = "k4v2"          # the default reference for a paired comparison

def Q(**kw):
    return ("compressed", QuantConfig(key_bits=4, value_bits=2, **kw))


# -- the rotation ablation -------------------------------------------------
#
# The claim: the rotation makes per-channel distributions Gaussian, so a fixed
# Lloyd-Max scalar codebook and one per-token RMS scale quantise well with no
# calibration data. Nothing in this file tested it. `rounds1`/`rounds3` above
# compare a rotation against a rotation; "three rounds is indistinguishable
# from two" says nothing about rotating versus not.
#
# Which published claim each arm tests
# ------------------------------------
# What this package stores is TurboQuant_mse (arXiv:2504.19874, Algorithm 1):
# rotate, then apply the Lloyd-Max scalar quantiser for the normal per
# coordinate, with the norm kept separately. The rotation is not a tweak on top
# of that method -- it is the premise the whole method rests on. Lemma 1 says a
# rotated unit vector has coordinates distributed as Beta, converging to
# N(0, 1/d); Theorem 1's distortion bound follows only because that is true. Take
# the rotation away and the theorem says nothing about what is left, which is why
# `rot0-gauss` is the arm theory predicts should be worst. A prior prediction to
# check the measurement against is worth more here than the measurement alone.
#
# But the paper proves that for a DENSE random orthogonal rotation, from the QR
# of a Gaussian. This package substitutes a randomized Hadamard, because O(d
# log d) is implementable in hardware and O(d^2) is not. That substitution is
# ours and is not covered by any proof in the paper. So the question these arms
# actually answer is not "does rotating help" -- that is settled for the dense
# case -- but "does the cheap structured stand-in deliver what the dense
# rotation is proved to deliver". The 8b/8b arms and `rotation-seeds` both read
# on that directly.
#
# It also raises what `uniform` is testing. Not merely "is the codebook shape
# worth its hardware", but the specific Lloyd-Max-for-N(0,1) choice Theorem 1
# depends on. If `rot1-unif` comes out level with `rot1-gauss`, the Gaussian
# table is not cashing in the distribution the rotation creates, which is a
# result about where the paper's argument applies at these bit-widths and not
# just about this implementation.
#
# Not tested here, and worth keeping straight: the paper's other quantiser,
# TurboQuant_prod -- MSE at b-1 bits plus a 1-bit QJL correction on the residual
# -- is not implemented anywhere in this package. Section 3.2 shows the MSE
# quantiser is biased for inner products, which is what keys are scored with.
# That is a separate known gap and no arm below moves it. The dense-versus-
# Hadamard comparison is not an arm either; `experiments/turboquant_attn.py`
# exists to measure that gap end to end and is the better instrument for it.
#
# Two things this has to get right to be worth running.
#
# The confound. `Codebook.gaussian` is optimal for the distribution the
# rotation *produces*. Removing the rotation while keeping that codebook
# compares a matched quantiser against a deliberately mismatched one, and would
# overstate the rotation. So every cell is a 2x2 over {rotation, no rotation} x
# {gaussian, uniform}: the honest number is rotation-off at its own best
# codebook, and the 2x2 splits the total into a rotation effect, a codebook
# effect, and their interaction. That interaction is the actual thesis -- the
# two should only pay off together.
#
# The falsification arm. 8b/8b is there to fail. At eight bits quantisation
# error should be small enough that the rotation buys close to nothing; if it
# still shows a large win there, the effect is not the distributional argument
# and something else is going on.
#
# Fixed in advance, so the result is not read backwards afterwards: rotation
# off, at its best codebook, must be worse than rotation on by more than two
# standard errors at 4b/2b, and the gap must shrink as width rises. Anything
# else gets written up as measured, including "the rotation buys nothing at
# these widths" -- which would be consistent with what ops/rotate.py already
# records about kurtosis failing to predict perplexity.

ROT_ARMS = {                     # arm name -> (rot_rounds, codebook)
    "rot1-gauss": (1, "gaussian"),      # what ships; the per-width reference
    "rot0-gauss": (0, "gaussian"),      # rotation removed, quantiser unchanged
    "rot0-unif":  (0, "uniform"),       # rotation removed, quantiser at its best
    "rot1-unif":  (1, "uniform"),       # separates rotation from codebook shape
}
ROT_REF_ARM = "rot1-gauss"

ROTATION = {"baseline": (None, None)}
for _kb, _vb in ((4, 2), (4, 4), (8, 8)):
    for _arm, (_rounds, _cb) in ROT_ARMS.items():
        ROTATION[f"{_kb}b{_vb}b-{_arm}"] = (
            "compressed", QuantConfig(key_bits=_kb, value_bits=_vb,
                                      rot_rounds=_rounds, codebook=_cb))

# Is the gain a property of the transform or of one lucky sign diagonal?
#
# Only the rotating arm varies here. At `rot_rounds=0` the sign diagonal is
# empty, so `seed` changes nothing and five no-rotation runs would be five
# copies of the same numbers -- one shared control, and every diagonal paired
# against it. Spread across seeds comparable to the gain itself would mean the
# headline measured a diagonal rather than the rotation.
ROTATION_SEEDS = {
    "baseline":  (None, None),
    "rot0-unif": ("compressed", QuantConfig(key_bits=4, value_bits=2,
                                            rot_rounds=0, codebook="uniform")),
    **{f"s{_s}-rot1-gauss": Q(seed=_s) for _s in range(5)},
}

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
    "rotation": ROTATION,
    "rotation-seeds": ROTATION_SEEDS,
}

# Which variant each one is paired against. A group spanning several bit widths
# cannot use one reference for all of it -- differencing an 8b/8b arm against a
# 4b/2b reference measures the width, not the thing being ablated -- so
# "rotation" pairs within a width, by the `4b2b-` prefix on every arm name.
REFS = {"rotation": "by-prefix", "rotation-seeds": "rot0-unif"}


def ref_name(group: str, name: str) -> str | None:
    """The reference for `name`, or None if it has none (the baseline)."""
    ref = REFS.get(group, REF)
    if ref != "by-prefix":
        return ref
    if "-" not in name:
        return None                          # "baseline"
    return f"{name.split('-', 1)[0]}-{ROT_REF_ARM}"


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
    ap.add_argument("--model", default=str(
        Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B"))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--dir", default="kernel/experiments/results/ablate")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # The model goes in the directory name. `results.json` is a resume cache
    # keyed on variant name alone, so a group re-run under a different model
    # would silently pair new variants against a cached reference from the old
    # one -- a paired comparison across two models, reported as if it were one.
    # That has already happened once, on the first rotation run.
    d = Path(args.dir) / f"{args.group}-c{args.context}-{Path(args.model).name.lower()}"
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
    base = state.get("baseline")
    print(f"\n{'variant':<18} {'ppl':>9} {'vs base':>9} {'vs ref':>10} "
          f"{'+- s.e.':>9} {'signif':>7} {'B/tok/L':>8} {'ref':>16}")
    for name, r in state.items():
        v = r["logppl"]
        ppl = float(np.exp(np.mean(v)))
        vb = f"{ppl / float(np.exp(np.mean(base['logppl']))):8.4f}x" if base else "        -"
        rn = ref_name(group, name)
        ref = state.get(rn) if rn else None
        if ref and name != rn:
            m, se = paired(v, ref["logppl"])
            ratio, sig = math.exp(m), ("yes" if abs(m) > 2 * se else "no")
            vr, ve = f"{ratio:9.4f}x", f"{se:9.4f}"
        elif name == rn:
            vr, ve, sig, rn = "    (ref)", "        -", "-", ""
        else:
            vr, ve, sig, rn = "        -", "        -", "-", ""
        print(f"{name:<18} {ppl:9.4f} {vb} {vr} {ve} {sig:>7} "
              f"{str(r.get('bytes_per_token_per_layer') or '-'):>8} {rn or '-':>16}")
    print(f"\npaired over {n_windows} windows; 'signif' = |difference| > 2 standard errors.")
    print("'vs ref' is the perplexity ratio against the variant named in 'ref'.")


if __name__ == "__main__":
    main()
