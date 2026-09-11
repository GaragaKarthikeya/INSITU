"""Independent re-check of the fp16 baseline perplexity, with zero imports
from this repo's `kernel` package -- only stock `transformers`, so a bug in
the kernel's own baseline path (ablate.py, ppl_paths.py) can't hide here.

Matches ablate.py's baseline measurement: WikiText-2 test split, 2,048-token
non-overlapping windows, mean per-token NLL exponentiated.

    /home/digital3/TurboQuant-Reproduction/.venv/bin/python \\
        -m kernel.experiments.verify_baseline_ppl \\
        --model models/Llama-3.2-1B --context 2048 --windows 4
"""

from __future__ import annotations

import argparse
import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_windows(tok, context: int, n_windows: int) -> list[torch.Tensor]:
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = context * n_windows
    if ids.numel() < need:
        raise ValueError(f"only {ids.numel()} tokens available, need {need}")
    return [ids[i * context:(i + 1) * context] for i in range(n_windows)]


@torch.no_grad()
def window_logppl(model, w: torch.Tensor) -> float:
    ids = w.unsqueeze(0)
    return float(model(ids, labels=ids).loss)  # mean NLL per token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    wins = load_windows(tok, args.context, args.windows)
    logppls = [window_logppl(model, w) for w in wins]
    mean_logppl = sum(logppls) / len(logppls)
    ppl = math.exp(mean_logppl)

    print(f"{args.model}  ctx={args.context}  windows={len(wins)}")
    for i, lp in enumerate(logppls):
        print(f"  window {i}: logppl={lp:.4f}  ppl={math.exp(lp):.4f}")
    print(f"\nmean logppl = {mean_logppl:.4f}")
    print(f"perplexity  = {ppl:.4f}")
    print("plan.MD's fp16 baseline: 7.1439")


if __name__ == "__main__":
    main()
