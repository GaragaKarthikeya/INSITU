"""Board vectors from a REAL Llama 3.2 1B, on real text.

WHY THIS IS NOT `board_vectors.py --seed 1`
-------------------------------------------
Every number the silicon has been fed so far came from
`rng.standard_normal(...) * 0.02`. That was the right stimulus for proving the
datapath: it is reproducible from a seed, it has no structure to hide a bug
behind, and it made twelve testbenches and 32 decode steps mean something.

It is the wrong stimulus for one question. **Real activations have a dynamic
range that pseudo-random ones do not.** `plan.MD`'s step 4 gate measured
exactly that -- 8,188 tokens of WikiText-2 through all 16 layers, peak
|x| = 24.957 against a +-128 rail, 19% above what 400 tokens had suggested --
and then every hardware run since has been on Gaussian noise at scale 0.02.

So the board's `clips`, `overflows`, `range_error` and `norm_saturated`
counters have never seen a real distribution. They have reported zero for 32
steps of stimulus chosen to be well-behaved. This is what makes them mean
something: the same counters, on the activations the model actually produces,
at a layer of a real checkpoint.

HOW THE HIDDEN STATES ARE CAPTURED
----------------------------------
The model runs UNGRAFTED and a spy records what arrives at one layer's
attention. Those are the true hidden states -- the ones the other fifteen
layers produced -- and they are then replayed through `AttentionKernel` by the
same `collect()` the testbenches use. Nothing here re-implements attention;
it substitutes real inputs into machinery that is already checked.

    python -m kernel.hw.real_vectors --layer 10 --steps 4 --out sw
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from .board_vectors import IN_WORDS, OUT_WORDS, _u32, emit
from .ddr_layout import DdrLayout
from .vectors import collect, pack_lanes

# Absolute, resolved from this file. A relative path works only when the
# process happens to start in the repo root, and transformers reports the
# failure as "not a valid model identifier on huggingface.co" -- which reads
# like a missing download rather than a wrong working directory.
MODEL = str(pathlib.Path(__file__).resolve().parents[1] / "models" / "Llama-3.2-1B")
PROMPT = ("The history of computing hardware spans centuries, from mechanical "
          "calculators through vacuum tubes and transistors to the integrated "
          "circuits that carry every modern processor. Each generation traded "
          "one scarce resource for another, and the memory wall is the trade "
          "that defines the present one.")

# The layers step 4's gate sampled across: early, middle, late. The peak
# activation is layer-dependent, so one layer is one sample and not a bound.
LAYERS = (2, 10, 15)


def wikitext(chars: int = 24000) -> str:
    """Real prose, the corpus `plan.MD` step 4's saturation gate used.

    Falls back to `PROMPT` when the dataset is not cached, because a demo that
    cannot run offline is a demo that cannot run.
    """
    try:
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "".join(d["text"])
        return text[:chars]
    except Exception as e:                                # noqa: BLE001
        print(f"  (wikitext unavailable: {e}; using the built-in prompt)")
        return PROMPT


def capture(layers, prompt: str = PROMPT, model_path: str = MODEL,
            max_tokens: int = 1024):
    """Real hidden states arriving at each layer's attention, one row per token.

    Every layer in one forward pass: the model is 2.5 GB and loading it four
    times to read four tensors would be the slowest possible way to do this.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()

    box, saved = {}, {}
    for L in layers:
        lyr = model.model.layers[L]
        saved[L] = lyr.self_attn.forward

        def spy(hidden_states, __L=L, __orig=saved[L], **kw):
            box[__L] = hidden_states.detach().clone()
            return __orig(hidden_states, **kw)

        lyr.self_attn.forward = spy
    try:
        ids = tok(prompt, return_tensors="pt")["input_ids"][:, :max_tokens]
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for L, f in saved.items():
            model.model.layers[L].self_attn.forward = f

    return ({L: box[L][0].numpy().astype(np.float32) for L in layers},
            model.config)


def case_from_hidden(x, hf, layer: int, steps: int = 4, seed: int = 0,
                     key_bits: int = 4, value_bits: int = 2) -> dict:
    """Build one board case from already-captured hidden states."""
    return _case(x, hf, layer, steps, seed, key_bits, value_bits)


def build_real_case(layer: int = 10, steps: int = 4, seed: int = 0,
                    key_bits: int = 4, value_bits: int = 2,
                    prompt: str = PROMPT, model_path: str = MODEL) -> dict:
    """The last `steps` captured tokens are decode steps; the rest is the cache."""
    from ..config import KernelConfig, ModelConfig, QuantConfig
    from ..kernel import AttentionKernel

    xs, hf = capture([layer], prompt, model_path)
    return _case(xs[layer], hf, layer, steps, seed, key_bits, value_bits)


def _case(x, hf, layer: int, steps: int, seed: int,
          key_bits: int, value_bits: int) -> dict:
    from ..config import KernelConfig, ModelConfig, QuantConfig
    from ..kernel import AttentionKernel

    n = x.shape[0]
    if n <= steps:
        raise ValueError(f"the prompt is {n} tokens, which leaves nothing to cache")
    ctx0 = n - steps

    m = ModelConfig(hidden_size=hf.hidden_size, num_heads=hf.num_attention_heads,
                    num_kv_heads=hf.num_key_value_heads,
                    head_dim=hf.hidden_size // hf.num_attention_heads,
                    max_position=max(4096, 2 * n + 2))
    cfg = KernelConfig(model=m, quant=QuantConfig(key_bits=key_bits,
                                                  value_bits=value_bits, seed=seed))
    k = AttentionKernel(cfg, capacity=n)
    k.forward(x[:ctx0])

    lay = DdrLayout.for_quant(cfg.quant, m.head_dim, m.num_kv_heads, n)
    image = lay.image(k.cache.buf[:ctx0])

    ingress, golden, n_tokens = [], [], []
    peak = 0.0
    for i in range(steps):
        n_tokens.append(k.cache.length + 1)
        vs = collect(k, x[ctx0 + i:ctx0 + i + 1])
        ingress.append(_u32(vs.ingress_bytes()))
        golden.append(_u32(pack_lanes(vs.out_online, vs.lane_bits)))
        peak = max(peak, float(np.abs(x[ctx0 + i]).max()))

    for h in range(m.num_kv_heads):
        for t in range(ctx0, n):
            if lay.row_at(image, h, t).any():
                raise AssertionError(f"head {h} token {t} is in the image")

    d = lay.describe()
    return {
        "ctx0": ctx0, "steps": steps, "capacity": n,
        "head_stride": d["head_stride"], "plane_span": d["plane_span"][0],
        "n_kv_heads": d["n_kv_heads"], "row_bytes": d["row_bytes"],
        "image": image, "ingress": np.concatenate(ingress),
        "golden": np.concatenate(golden),
        "n_tokens": np.array(n_tokens, dtype="<u4"),
        "peak_abs": peak, "layer": layer, "tokens": n,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", default="", help="default: 2,10,15")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--key-bits", type=int, default=4)
    ap.add_argument("--value-bits", type=int, default=2)
    ap.add_argument("--prompt", action="store_true",
                    help="use the built-in prompt instead of WikiText")
    ap.add_argument("--out", default="sw")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)

    layers = [int(v) for v in a.layers.split(",")] if a.layers else list(LAYERS)
    text = PROMPT if a.prompt else wikitext()
    print(f"capturing layers {layers} from {a.model}")
    xs, hf = capture(layers, text, a.model, max_tokens=a.tokens)

    cases = []
    for L in layers:
        c = _case(xs[L], hf, L, a.steps, 0, a.key_bits, a.value_bits)
        c["name"] = f"realL{L}"
        cases.append(c)
        print(f"  layer {L:2d}: {c['tokens']} real tokens, {c['ctx0']} cached, "
              f"{c['steps']} steps, peak |x| {c['peak_abs']:.3f} "
              f"(the Q8.16 rail is 128), image {c['image'].size} B")
    paths = emit(cases, a.out)
    print("wrote " + ", ".join(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
