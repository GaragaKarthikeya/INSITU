"""Which side of the cache is expensive: the keys or the values?

The headline sweep moves both widths at once, so it cannot say. This holds one
fixed and moves the other. It matters for the format: the two planes need not
be the same width, and if one is cheap it should be narrowed first.
"""
import sys
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernel import ModelConfig, QuantConfig
from kernel.adapters.torch_llama import graft, ungraft
from kernel.experiments.full_model import M, TEXT, perplexity


def main(n_tokens: int = 400):
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :n_tokens]
    layers = list(range(model.config.num_hidden_layers))
    m = ModelConfig.from_hf(model.config)
    base = perplexity(model, ids)
    print(f"baseline ppl {base:.3f} over {ids.shape[1]} tokens, "
          f"{len(layers)} layers grafted\n")

    def run(kb, vb):
        original = graft(model, layers, quant=QuantConfig(key_bits=kb, value_bits=vb),
                         capacity=n_tokens + 8)
        try:
            ppl = perplexity(model, ids)
            bpt = model.model.layers[0].self_attn.kernel.cache.bytes_per_token
        finally:
            ungraft(model, original)
        return ppl, bpt

    print("keys swept, values fixed at 4b")
    print(f"{'key bits':>9} {'ppl':>8} {'vs base':>9} {'B/tok':>7}")
    for kb in (6, 5, 4, 3, 2):
        ppl, bpt = run(kb, 4)
        print(f"{kb:9d} {ppl:8.3f} {ppl / base:8.3f}x {bpt:7d}")

    print("\nvalues swept, keys fixed at 4b")
    print(f"{'val bits':>9} {'ppl':>8} {'vs base':>9} {'B/tok':>7}")
    for vb in (6, 5, 4, 3, 2):
        ppl, bpt = run(4, vb)
        print(f"{vb:9d} {ppl:8.3f} {ppl / base:8.3f}x {bpt:7d}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
