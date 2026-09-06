"""Step 2: graft every layer of a real TinyLlama and measure what it costs.

Perplexity on real text, baseline against the kernel at several code widths.
The dense-mode row is the control: it exercises the whole kernel -- projection,
the fp->fixed cast, RoPE, the rotation, the fold -- with no quantisation, so
any gap in that row is the kernel's own fixed-point error and everything below
it is attributable to the compression.
"""
import sys, time
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernel import ModelConfig, QuantConfig
from kernel.adapters.torch_llama import graft, ungraft

M = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TEXT = """The history of computing hardware spans centuries, from mechanical
calculators to the integrated circuits that power modern devices. Charles
Babbage designed the Analytical Engine in the 1830s, a mechanical
general-purpose computer that was never fully built in his lifetime. Ada
Lovelace wrote what is considered the first algorithm intended for
implementation on such a machine. In the twentieth century, vacuum tubes gave
way to transistors, and transistors gave way to integrated circuits, each
transition bringing exponential improvements in speed, density, and cost.
Modern processors contain billions of transistors on a single die, fabricated
with features measured in nanometres. The memory hierarchy that surrounds them
has grown just as elaborate: registers, several levels of cache, and main
memory, each an order of magnitude slower and larger than the one above it.

The economics of this hierarchy shape almost every design decision. A value
held in a register costs nothing to read; a value fetched from main memory
costs hundreds of cycles, during which the processor may have nothing useful
to do. Architects therefore spend enormous effort predicting which values will
be needed, prefetching them early, and arranging computations so that data
already close to the arithmetic units is reused as much as possible before it
is evicted. The same pressure appears again in specialised accelerators, where
the arithmetic is cheap and abundant but the bandwidth feeding it is neither.

Language models make this tension unusually stark. Generating a single token
requires reading every weight in the network exactly once, and reusing none of
them, so the arithmetic units sit mostly idle while the memory system works at
full stretch. Batching many requests together restores the reuse and the
efficiency, but it also raises latency, and interactive applications cannot
always afford to wait. The attention mechanism adds a second and growing
demand: the keys and values of every previous token must be read again for
every new one, so the cost of a conversation rises with its length. Compressing
that cache is therefore not a micro-optimisation but a structural change to
what the machine spends its time doing, and whether it pays depends on details
that only a careful measurement can settle."""


def perplexity(model, ids) -> float:
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss))


def main(n_tokens: int = 160):
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :n_tokens]
    n = ids.shape[1]

    m = ModelConfig.from_hf(model.config)
    layers = list(range(model.config.num_hidden_layers))
    print(f"model      TinyLlama-1.1B, {len(layers)} layers grafted, {n} tokens")
    print(f"geometry   hidden {m.hidden_size}, {m.num_heads} heads, "
          f"{m.num_kv_heads} kv heads, d_head {m.head_dim}\n")

    t0 = time.time()
    base = perplexity(model, ids)
    print(f"{'configuration':<28} {'ppl':>8} {'vs base':>9} {'B/tok/layer':>12} "
          f"{'vs fp16':>8} {'time':>7}")
    print(f"{'baseline (real fp32 attn)':<28} {base:8.3f} {'--':>9} "
          f"{2 * 2 * m.head_dim * m.num_kv_heads:12d} {'1.00x':>8} "
          f"{time.time() - t0:6.1f}s")

    configs = [
        ("kernel, dense (control)", QuantConfig(), False),
        ("key 6b / value 6b", QuantConfig(key_bits=6, value_bits=6), True),
        ("key 4b / value 4b", QuantConfig(key_bits=4, value_bits=4), True),
        ("key 4b / value 3b", QuantConfig(key_bits=4, value_bits=3), True),
        ("key 3b / value 2b", QuantConfig(key_bits=3, value_bits=2), True),
        ("key 2b / value 2b", QuantConfig(key_bits=2, value_bits=2), True),
    ]
    dense_bpt = 2 * 2 * m.head_dim * m.num_kv_heads

    for label, quant, compressed in configs:
        t0 = time.time()
        original = graft(model, layers, quant=quant, capacity=n + 8,
                         compressed=compressed)
        try:
            ppl = perplexity(model, ids)
            bpt = model.model.layers[0].self_attn.kernel.cache.bytes_per_token
        finally:
            ungraft(model, original)
        print(f"{label:<28} {ppl:8.3f} {ppl / base:8.3f}x {bpt:12d} "
              f"{dense_bpt / bpt:7.2f}x {time.time() - t0:6.1f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 160)
