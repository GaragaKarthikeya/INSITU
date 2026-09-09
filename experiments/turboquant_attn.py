"""TurboQuant_mse as an attention block, in float, graftable into a real model.

    python -m kernel.experiments.turboquant_attn --tokens 2048 --bits 4

This is the published method end to end, not this project's version of it: a
dense random rotation, an L2 norm kept in floating point, centroids scaled by
1/sqrt(d), and scores taken by reconstructing the key first, exactly as the
paper's `<y, Q^-1(Q(x))>` formulation says. Everything is float64.

It exists to separate two costs that are easy to conflate. We already know the
fixed-point arithmetic is free -- the fused path reproduces float64 perplexity
to four decimal places. So whatever gap appears between this and the kernel is
the FORMAT: the Hadamard substituted for the dense rotation, and the norm
quantised to 16 bits.

The reconstruction is done once for the whole cache rather than once per query.
That is an evaluation strategy and not a change to the method -- dequantisation
is deterministic, so reconstructing a key once and reusing it is the identical
number, and doing it per query would make a 2048-token run take hours.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernel.config import ModelConfig                                  # noqa: E402
from kernel.experiments.turboquant_ref import TurboQuantMSE            # noqa: E402
from kernel.ops.rope import RoPE                                       # noqa: E402


class _Cfg:
    def __init__(self, model):
        self.model = model


class TurboQuantKernel:
    """`forward(hidden, positions) -> (y, report)`, so `KernelAttention` fits."""

    def __init__(self, attn_module, model_config, bits_k=4, bits_v=4,
                 capacity=4096, seed=0):
        m = ModelConfig.from_hf(model_config)
        self.cfg = _Cfg(m)
        d = m.head_dim

        def w(name):
            p = getattr(attn_module, name, None)
            return None if p is None else p.weight.detach().cpu().double().numpy()

        def b(name):
            p = getattr(attn_module, name, None)
            bb = None if p is None else getattr(p, "bias", None)
            return None if bb is None else bb.detach().cpu().double().numpy()

        self.wq, self.wk, self.wv, self.wo = (w("q_proj"), w("k_proj"),
                                              w("v_proj"), w("o_proj"))
        self.bq, self.bk, self.bv, self.bo = (b("q_proj"), b("k_proj"),
                                              b("v_proj"), b("o_proj"))

        self.rope = RoPE.build(d, m.rope_theta, m.max_position,
                               scaling=m.rope_scaling)
        self.tq_k = TurboQuantMSE(d, bits_k, seed=seed)
        self.tq_v = TurboQuantMSE(d, bits_v, seed=seed + 1)

        self.capacity = capacity
        self.k_codes = np.zeros((capacity, m.num_kv_heads, d), dtype=np.uint8)
        self.v_codes = np.zeros((capacity, m.num_kv_heads, d), dtype=np.uint8)
        self.k_norm = np.zeros((capacity, m.num_kv_heads), dtype=np.float64)
        self.v_norm = np.zeros((capacity, m.num_kv_heads), dtype=np.float64)
        self.length = 0

    def reset(self):
        self.length = 0

    def forward(self, hidden_states, positions=None):
        m = self.cfg.model
        d, H, KVH = m.head_dim, m.num_heads, m.num_kv_heads
        g = H // KVH

        x = np.asarray(hidden_states, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        if positions is None:
            positions = np.arange(self.length, self.length + n)
        positions = np.asarray(positions, dtype=np.int64)

        def proj(w, bias, heads):
            y = x @ w.T
            if bias is not None:
                y = y + bias
            return y.reshape(n, heads, d)

        q = proj(self.wq, self.bq, H)
        k = proj(self.wk, self.bk, KVH)
        v = proj(self.wv, self.bv, KVH)

        q = self.rope.apply_float(q, positions[:, None])
        k = self.rope.apply_float(k, positions[:, None])

        # -- quantize and store, one TurboQuant vector per token per KV head
        base = self.length
        for h in range(KVH):
            ck, nk = self.tq_k.quantize(k[:, h])
            cv, nv = self.tq_v.quantize(v[:, h])
            self.k_codes[base:base + n, h] = ck
            self.v_codes[base:base + n, h] = cv
            self.k_norm[base:base + n, h] = nk
            self.v_norm[base:base + n, h] = nv
        self.length = base + n
        T = self.length

        # -- attend, reconstructing first, as the paper specifies
        out = np.zeros((n, H, d), dtype=np.float64)
        scale = 1.0 / math.sqrt(d)
        valid = np.arange(T)[None, :] <= (base + np.arange(n))[:, None]
        for h in range(KVH):
            k_hat = self.tq_k.dequantize(self.k_codes[:T, h], self.k_norm[:T, h])
            v_hat = self.tq_v.dequantize(self.v_codes[:T, h], self.v_norm[:T, h])
            qh = q[:, h * g:(h + 1) * g]                       # (n, g, d)
            s = np.einsum("ngd,td->ngt", qh, k_hat) * scale
            s = np.where(valid[:, None, :], s, -np.inf)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            out[:, h * g:(h + 1) * g] = np.einsum("ngt,td->ngd", p, v_hat)

        y = out.reshape(n, H * d) @ self.wo.T
        if self.bo is not None:
            y = y + self.bo
        return y, None


def graft_turboquant(model, layers, bits_k, bits_v, capacity, seed=0):
    """Replace each layer's attention with the float TurboQuant block."""
    from kernel.adapters.torch_llama import KernelAttention

    original = {}
    for i in layers:
        mod = model.model.layers[i].self_attn
        original[i] = mod
        kern = TurboQuantKernel(mod, model.config, bits_k=bits_k, bits_v=bits_v,
                                capacity=capacity, seed=seed)
        model.model.layers[i].self_attn = KernelAttention(kern, layer_idx=i)
    return original


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--bits", type=int, default=4, help="both K and V")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kernel.adapters.torch_llama import ungraft
    from kernel.experiments.fpga_infer import MODEL
    from kernel.experiments.fpga_ppl import CHARS_PER_TOKEN, wikitext
    from kernel.experiments.ppl_paths import nll

    print(f"loading {MODEL} (fp32, cpu)")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    text = wikitext(max(8000, a.tokens * CHARS_PER_TOKEN))
    ids = tok(text, return_tensors="pt")["input_ids"][0][:a.tokens]
    n_layers = model.config.num_hidden_layers
    print(f"{ids.shape[0]} tokens of WikiText-2, {n_layers} layers, "
          f"TurboQuant_mse {a.bits}b/{a.bits}b\n")

    base = math.exp(sum(nll(model, ids)) / (ids.shape[0] - 1))
    original = graft_turboquant(model, list(range(n_layers)), a.bits, a.bits,
                                int(ids.shape[0]) + 1, a.seed)
    try:
        ppl = math.exp(sum(nll(model, ids)) / (ids.shape[0] - 1))
    finally:
        ungraft(model, original)

    print(f"    {'path':<40}{'perplexity':>12}{'vs baseline':>14}")
    print(f"    {'baseline, unmodified model':<40}{base:>12.4f}{'—':>14}")
    print(f"    {'TurboQuant_mse, float, dense rotation':<40}"
          f"{ppl:>12.4f}{ppl / base:>13.4f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
