"""Step 3: actually generate text through the grafted kernel.

Perplexity is a number; generation is the thing that breaks loudly when a
position, a head layout or a cache is wrong. This runs greedy decoding, which
exercises the path perplexity does not: prefill, then one token at a time, with
the cache growing under it.
"""
import sys, time
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernel import QuantConfig
from kernel.adapters.torch_llama import graft, ungraft

M = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = ("<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n"
          "In one short paragraph, explain what a KV cache is in a transformer."
          "</s>\n<|assistant|>\n")


def reset_all(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "reset"):
            attn.reset()


@torch.no_grad()
def greedy(model, tok, prompt, n_new=48):
    """Hand-rolled greedy loop: prefill once, then one token at a time.

    Written out rather than calling `model.generate` so the decode path is
    unambiguous -- `generate` has its own cache handling, and the point here is
    to drive the kernel's cache the way a serving loop does.
    """
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model(ids, use_cache=True)
    past = out.past_key_values
    token = out.logits[:, -1].argmax(-1, keepdim=True)
    produced = [token.item()]

    for _ in range(n_new - 1):
        out = model(token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        produced.append(token.item())
        if token.item() == tok.eos_token_id:
            break
    return tok.decode(produced, skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    layers = list(range(model.config.num_hidden_layers))

    t0 = time.time()
    print("=== BASELINE (real fp32 attention) ===")
    print(greedy(model, tok, PROMPT).strip())
    print(f"[{time.time() - t0:.1f}s]\n")

    for label, quant in [
        ("key 4b / value 3b  (4.27x)", QuantConfig(key_bits=4, value_bits=3)),
        ("key 3b / value 2b  (5.82x)", QuantConfig(key_bits=3, value_bits=2)),
    ]:
        t0 = time.time()
        original = graft(model, layers, quant=quant, capacity=512)
        try:
            reset_all(model)
            print(f"=== GRAFTED KERNEL, {label} ===")
            print(greedy(model, tok, PROMPT).strip())
        finally:
            ungraft(model, original)
        print(f"[{time.time() - t0:.1f}s]\n")


if __name__ == "__main__":
    main()
