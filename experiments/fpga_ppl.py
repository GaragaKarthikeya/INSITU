"""Perplexity with every layer's attention on the FPGA, against the same model.

    python -m kernel.experiments.fpga_ppl --eth enp4s0 --tokens 512

What this measures, and what it does not
----------------------------------------
Two numbers on the same text and the same checkpoint:

  * `baseline` -- the unmodified model, batched, on the host.
  * `fpga` -- every layer's attention computed on the ZCU104, one token at a
    time, with all sixteen KV caches built by the block's own write master.

The ratio is what the cache compression costs, end to end, with the hardware in
the loop. It is not a claim about the FPGA versus numpy: those are checked
per-step for bit-exactness and any difference there is a hardware bug, reported
separately. It is a claim about 4-bit keys and 2-bit values.

Why it is slow, and why that is fine
------------------------------------
Every token costs sixteen board round trips, so a 512-token passage is 8,192 of
them. Over raw Ethernet that is seconds. Over JTAG it is half an hour, which is
why `--eth` is not optional here in practice.

The baseline runs batched because it has no reason not to -- it is the
reference, not the measurement.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.experiments.fpga_infer import CACHE_BASE, MODEL, open_client  # noqa: E402
from kernel.host.fpga_layers import FpgaLayers                            # noqa: E402


# WikiText-2 test is about 1.2 MB, so the only real cap is how much of it is
# asked for. The join is `""`, not `"\n\n"`: the rows already carry their own
# newlines and joining them differently would move the baseline perplexity,
# which is the number every earlier result is quoted against.
CHARS_PER_TOKEN = 8      # English is ~4; double it so the slice is never short


def wikitext(n_chars: int = 8000) -> str:
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "".join(d["text"])[:n_chars]


def baseline_nll(model, ids) -> list[float]:
    """The unmodified model, batched. `ids` is a 1-D tensor of token ids."""
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
    ap.add_argument("--udp", metavar="IP", nargs="?", const="192.168.10.2")
    ap.add_argument("--eth", metavar="IFACE")
    ap.add_argument("--mac", default="02:00:5a:77:e0:01")
    ap.add_argument("--jtag", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-step comparison against numpy. The "
                         "perplexity is unaffected -- the board's numbers are "
                         "what the model consumes either way -- and this is "
                         "the difference between a long corpus taking minutes "
                         "and taking hours.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = ("cuda" if (a.device == "auto" and torch.cuda.is_available())
           else ("cpu" if a.device == "auto" else a.device))
    print(f"loading {a.model} onto {dev} ({a.dtype})")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model,
                                                 dtype=getattr(torch, a.dtype))
    model.eval().to(dev)

    # Enough text for the token budget, rather than a fixed 8,000 characters.
    #
    # The slice used to be a constant, which is about 1,900 Llama tokens. Asking
    # for more did not fail -- it silently measured a shorter passage, so a
    # 4,096-token run and a 2,048-token run would have returned the same
    # perplexity over the same text and nothing would have said so. The
    # assertion below is the part that matters: a run reports the length it was
    # asked for or it does not run.
    text = wikitext(max(8000, a.tokens * CHARS_PER_TOKEN))
    ids = tok(text, return_tensors="pt")["input_ids"][0][:a.tokens].to(dev)
    if ids.shape[0] < a.tokens:
        raise SystemExit(
            f"asked for {a.tokens} tokens, the corpus yielded {ids.shape[0]}. "
            f"WikiText-2 test is finite; raise CHARS_PER_TOKEN if the slice is "
            f"the limit, or lower --tokens if the split is.")
    print(f"{ids.shape[0]} tokens of WikiText-2 "
          f"({len(text)} characters read)")

    # -- the reference, on the unmodified model ---------------------------
    print("baseline (unmodified, batched)")
    base = baseline_nll(model, ids)
    base_ppl = math.exp(sum(base) / len(base))
    print(f"  perplexity {base_ppl:.3f}")

    # -- the same text, every layer on the board --------------------------
    n_layers = model.config.num_hidden_layers
    capacity = int(ids.shape[0]) + 1
    client = open_client(a)
    if client is None:
        from kernel.adapters.torch_llama import graft
        graft(model, layers=list(range(n_layers)), capacity=capacity)
        fpga = None
        print("--cpu-only: compressed attention in numpy, board untouched")
    else:
        fpga = FpgaLayers(model, client, list(range(n_layers)), capacity,
                          cache_base=CACHE_BASE, verify=not a.no_verify,
                          quiet=True)
        print(f"{n_layers} layers on the board, {fpga.bytes_used / 1e6:.1f} MB "
              f"of DDR")

    nll = []
    t0 = time.time()
    past = None
    with torch.no_grad():
        for i in range(ids.shape[0]):
            out = model(input_ids=ids[i].view(1, 1), past_key_values=past,
                        use_cache=True)
            past = out.past_key_values
            if i + 1 < ids.shape[0]:
                logp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
                nll.append(-float(logp[int(ids[i + 1])]))
            if (i + 1) % 32 == 0:
                run = math.exp(sum(nll) / max(len(nll), 1))
                print(f"  {i + 1}/{ids.shape[0]} tokens, running ppl {run:.3f}, "
                      f"{time.time() - t0:.0f} s")
    wall = time.time() - t0
    ppl = math.exp(sum(nll) / len(nll))

    print()
    print("=" * 72)
    print(f"baseline (fp32 attention, host)      perplexity {base_ppl:8.3f}")
    print(f"compressed 4b/2b, {'FPGA' if fpga else 'numpy'}"
          f"{'':<15} perplexity {ppl:8.3f}")
    print(f"                                     ratio      {ppl / base_ppl:8.4f}")
    print("=" * 72)
    print(f"{ids.shape[0]} tokens in {wall:.1f} s "
          f"({wall / ids.shape[0] * 1000:.0f} ms/token wall)")
    if fpga:
        print(fpga.summary())
        if hasattr(client, "quit"):
            client.quit()
        if fpga.steps == 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
