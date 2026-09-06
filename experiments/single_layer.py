"""Step 1: does the kernel reproduce one real attention layer?

The smoke test that must pass before any accuracy number means anything. It
runs the real `LlamaAttention` and the kernel on the same captured hidden
states, with the kernel in dense mode (no quantisation) so that any residual is
plumbing -- RoPE convention, head layout, the W_o fold -- and not compression.

Then it repeats in compressed mode, where the residual is the compression.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernel import AttentionKernel, KernelConfig, ModelConfig, QuantConfig, Weights
from kernel.ops.rotate import Rotation

M = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TEXT = ("The history of computing hardware spans centuries, from mechanical "
        "calculators to the integrated circuits that power modern devices. "
        "Charles Babbage designed the Analytical Engine in the 1830s.")


def capture(model, tok):
    """Real hidden states into layer LAYER's attention, and its real output."""
    box = {}
    layer = model.model.layers[LAYER]
    orig = layer.self_attn.forward

    def spy(hidden_states, **kw):
        out = orig(hidden_states, **kw)
        box["in"] = hidden_states.detach().clone()
        box["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        box["position_ids"] = kw.get("position_ids")
        return out

    layer.self_attn.forward = spy
    with torch.no_grad():
        model(**tok(TEXT, return_tensors="pt"))
    layer.self_attn.forward = orig
    return box


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def cos(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()
    box = capture(model, tok)

    m = ModelConfig.from_hf(model.config)
    attn = model.model.layers[LAYER].self_attn
    x = box["in"][0].numpy().astype(np.float32)
    ref = box["out"][0].numpy().astype(np.float32)
    pos = box["position_ids"].reshape(-1).numpy().astype(np.int64)

    print(f"layer {LAYER}, {x.shape[0]} tokens, hidden {x.shape[1]}")
    print(f"reference output: max |y| = {np.abs(ref).max():.4f}\n")

    def w(n):
        return getattr(attn, f"{n}_proj").weight.detach().numpy().astype(np.float64)

    def b(n):
        lin = getattr(attn, f"{n}_proj")
        return None if lin.bias is None else lin.bias.detach().numpy()

    def gain(n):
        mod = getattr(attn, n, None)
        return None if mod is None else mod.weight.detach().numpy()

    eps = getattr(getattr(attn, "q_norm", None), "variance_epsilon",
                  getattr(model.config, "rms_norm_eps", 1e-6))
    extras = dict(q_norm=gain("q_norm"), k_norm=gain("k_norm"), norm_eps=eps,
                  q_bias=b("q"), k_bias=b("k"), v_bias=b("v"), o_bias=b("o"))
    print(f"QK-norm: {extras['q_norm'] is not None}   "
          f"QKV bias: {extras['q_bias'] is not None}\n")

    rows = []
    for label, quant, compressed in [
        ("dense fp16 (plumbing only)", QuantConfig(), False),
        ("key 6b / value 6b", QuantConfig(key_bits=6, value_bits=6), True),
        ("key 4b / value 4b", QuantConfig(key_bits=4, value_bits=4), True),
        ("key 3b / value 2b", QuantConfig(key_bits=3, value_bits=2), True),
        ("key 2b / value 2b", QuantConfig(key_bits=2, value_bits=2), True),
    ]:
        rot = Rotation.for_quant(quant, m.head_dim)
        weights = Weights.prepare(w("q"), w("k"), w("v"), w("o"), m, rot, **extras)
        cfg = KernelConfig(model=m, quant=quant)
        k = AttentionKernel(cfg, weights, capacity=len(pos) + 8, compressed=compressed)
        y, rep = k.forward(x, pos)
        bpt = k.cache.bytes_per_token
        rows.append((label, rel(y, ref), cos(y, ref), bpt, rep))

    print(f"{'configuration':<28} {'rel err':>9} {'cosine':>9} {'B/token':>8} {'vs dense':>9}")
    dense_bpt = rows[0][3]
    for label, r, c, bpt, rep in rows:
        print(f"{label:<28} {r:9.2e} {c:9.6f} {bpt:8d} {dense_bpt / bpt:8.2f}x")

    print(f"\nfp->fixed saturation: {rows[0][4].conversion.saturation_rate:.3%} "
          f"(max |x| = {rows[0][4].conversion.max_abs_float:.3f})")


if __name__ == "__main__":
    main()
