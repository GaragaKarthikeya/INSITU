"""Drop the kernel into a Llama-family model in place of its attention block.

    from kernel.adapters.torch_llama import graft

    model = AutoModelForCausalLM.from_pretrained(...)
    graft(model, layers=[10], quant=QuantConfig(key_bits=3, value_bits=2))

What this is and is not
-----------------------
It is a torch `nn.Module` that holds a numpy `AttentionKernel`, converts at the
boundary, and presents the signature a decoder layer already calls. That is the
whole reason `AttentionKernel.forward` takes `(tokens, hidden)` and returns the
same: the adapter is thin because the kernel was shaped to be grafted.

It is not fast. Every call crosses torch -> numpy -> torch and the attention
loop is Python. It is for measuring what the compression does to a real model's
outputs, not for serving.

torch is imported lazily and is not a dependency of the package. Everything
outside this file runs on numpy alone.

The contract, as observed
-------------------------
Probed against transformers 5.16.1 / LlamaAttention
(`experiments/probe_contract.py`), a decoder layer calls its attention module
with:

    attention_mask, position_ids, past_key_values, use_cache, position_embeddings

and expects a `(output, attn_weights)` tuple back. Two consequences that a
stub test did not catch and running it did:

  * **`position_ids`, not `cache_position`.** Reading only the latter silently
    falls back to the adapter's own token counter, which happens to be right
    for sequential decoding and wrong for anything else.
  * **`past_key_values` must be advanced.** The model asks the cache object how
    long the context is; if the grafted layer never appends, that length stays
    zero and every position after the first is wrong. So the adapter writes its
    post-RoPE K/V into the passed cache for bookkeeping only -- it never reads
    it back, and the compressed cache the memory claims are about is the
    kernel's own. Grafting one layer hides this bug, because the other layers
    keep the count right; grafting all of them exposes it.

`forward` absorbs unknown keyword arguments, because this contract has changed
across transformers releases. Run `experiments/single_layer.py` after any
version bump: it checks the kernel in dense mode against the real
`LlamaAttention` on captured hidden states, where any residual is plumbing and
not compression. Measured at transformers 5.16.1, TinyLlama-1.1B layer 10:
**4.4e-4 relative, cosine 1.000000**.
"""

from __future__ import annotations

import numpy as np

from ..config import FixedFormat, KernelConfig, ModelConfig, QuantConfig
from ..kernel import AttentionKernel
from ..weights import Weights


class KernelAttention:
    """Adapter with the shape of a Llama attention module.

    Subclasses `torch.nn.Module` at construction time rather than at import
    time, so importing this file does not require torch.
    """

    def __new__(cls, *args, **kwargs):
        import torch.nn as nn
        if not issubclass(cls, nn.Module):
            cls = type(cls.__name__, (cls, nn.Module), {})
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def __init__(self, kernel: AttentionKernel, layer_idx: int = 0,
                 dtype=None, device=None) -> None:
        self.kernel = kernel
        self.layer_idx = layer_idx
        self._dtype = dtype
        self._device = device
        self.last_report = None

    # -- the decoder layer's contract --------------------------------------

    def forward(self, hidden_states, *args, **kwargs):
        """`(batch, seq, hidden)` in, `(output, None)` out.

        Extra positional and keyword arguments -- `position_embeddings`,
        `attention_mask`, `past_key_value`, `cache_position` -- are accepted and
        Ignored, and that is a real limitation, not an oversight:

          * RoPE is applied inside the kernel from its own table, so an
            externally supplied `position_embeddings` would be applied twice.
          * The KV cache is the kernel's own, so `past_key_value` is unused and
            the model's cache object will not reflect what is stored.
          * `attention_mask` is not honoured. The kernel is causal by
            construction, which covers decoding, but not padded batches or any
            custom mask. Grafting into a padded batch gives wrong answers
            silently, so `batch > 1` is refused below rather than tolerated.
        """
        import torch

        x = hidden_states
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        if x.shape[0] != 1:
            raise NotImplementedError(
                f"batch of {x.shape[0]}: this adapter holds one sequence's cache "
                f"and ignores attention_mask, so a padded batch would silently "
                f"attend across sequences. Run one sequence at a time."
            )

        # `position_ids` is what a Llama decoder layer actually passes;
        # `cache_position` is accepted as the older spelling. Falling through
        # to None uses the kernel's own counter, which is right only if every
        # call has been sequential from a reset.
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = kwargs.get("cache_position")
        if positions is not None:
            positions = positions.reshape(-1)
        pos = positions.detach().cpu().numpy().astype(np.int64) \
            if positions is not None else None

        arr = x[0].detach().cpu().float().numpy()
        y, report = self.kernel.forward(arr, pos)
        self.last_report = report

        self._advance_hf_cache(kwargs.get("past_key_values"), x)

        out = torch.from_numpy(np.ascontiguousarray(y)).to(
            device=x.device, dtype=self._dtype or hidden_states.dtype)
        return (out if squeeze else out.unsqueeze(0)), None

    __call__ = forward

    def _advance_hf_cache(self, cache, x) -> None:
        """Keep the model's own length bookkeeping correct. Bookkeeping only.

        The kernel never reads this back -- it has its own compressed cache, and
        that is the one every byte figure refers to. What is stored here exists
        so `past_key_values.get_seq_length()` returns the truth, because the
        model uses it to place the next token's position.
        """
        if cache is None:
            return
        import torch
        n_kv, d = self.kernel.cfg.model.num_kv_heads, self.kernel.cfg.model.head_dim
        shape = (x.shape[0], n_kv, x.shape[1], d)
        filler = torch.zeros(shape, dtype=x.dtype, device=x.device)
        try:
            cache.update(filler, filler, self.layer_idx)
        except Exception:
            # Cache APIs move between releases. A failure here is a bookkeeping
            # problem, not a numeric one, and must not be swallowed silently.
            raise RuntimeError(
                f"could not advance past_key_values ({type(cache).__name__}) for "
                f"layer {self.layer_idx}; positions after the first token would "
                f"be wrong. Pass past_key_values=None, or update this shim."
            )

    def reset(self) -> None:
        self.kernel.reset()


# --------------------------------------------------------------------------
# grafting
# --------------------------------------------------------------------------

def build_from_layer(attn_module, model_config, *, quant: QuantConfig | None = None,
                     fmt: FixedFormat | None = None, hw=None,
                     capacity: int = 4096, compressed: bool = True,
                     layer_idx: int = 0) -> KernelAttention:
    """Build a `KernelAttention` from an existing attention module's weights."""
    m = ModelConfig.from_hf(model_config)
    quant = quant or QuantConfig()
    cfg = KernelConfig(model=m, quant=quant, fmt=fmt or FixedFormat(), hw=hw)

    from ..ops.rotate import Rotation
    rotation = Rotation.for_quant(quant, m.head_dim)

    def w(name):
        return getattr(attn_module, f"{name}_proj").weight.detach().cpu().float().numpy()

    def b(name):
        layer = getattr(attn_module, f"{name}_proj")
        return None if layer.bias is None else layer.bias.detach().cpu().float().numpy()

    def gain(name):
        mod = getattr(attn_module, name, None)
        return None if mod is None else mod.weight.detach().cpu().float().numpy()

    eps = getattr(getattr(attn_module, "q_norm", None), "variance_epsilon",
                  getattr(model_config, "rms_norm_eps", 1e-6))
    weights = Weights.prepare(
        w("q"), w("k"), w("v"), w("o"), m, rotation,
        q_norm=gain("q_norm"), k_norm=gain("k_norm"), norm_eps=eps,
        q_bias=b("q"), k_bias=b("k"), v_bias=b("v"), o_bias=b("o"))
    kernel = AttentionKernel(cfg, weights, capacity=capacity, compressed=compressed)
    return KernelAttention(kernel, layer_idx=layer_idx)


def graft(model, layers, **kw) -> dict:
    """Replace `model.model.layers[i].self_attn` for each `i` in `layers`.

    Returns the modules that were replaced, so the graft can be undone -- which
    matters, because comparing grafted against original on the same model
    object is the only honest way to attribute a change to the kernel.
    """
    original = {}
    for i in layers:
        layer = model.model.layers[i]
        original[i] = layer.self_attn
        layer.self_attn = build_from_layer(
            layer.self_attn, model.config, layer_idx=i, **kw)
    return original


def ungraft(model, original: dict) -> None:
    for i, module in original.items():
        model.model.layers[i].self_attn = module


def compare_against_original(model, layer_idx: int, hidden_states, **kw):
    """Run one layer both ways on the same input and report the difference.

    The smoke test to run first after any graft. A large residual here means
    the plumbing is wrong -- a RoPE convention, a head layout, a missing fold
    -- and no accuracy conclusion drawn downstream would mean anything.
    """
    import torch

    layer = model.model.layers[layer_idx]
    with torch.no_grad():
        ref = layer.self_attn(hidden_states, **kw)
        ref = ref[0] if isinstance(ref, tuple) else ref

    adapter = build_from_layer(layer.self_attn, model.config, layer_idx=layer_idx)
    with torch.no_grad():
        got = adapter(hidden_states, **kw)[0]

    a = ref.detach().cpu().float().numpy()
    b = got.detach().cpu().float().numpy()
    denom = max(float(np.abs(a).max()), 1e-9)
    return {
        "max_abs": float(np.abs(a - b).max()),
        "relative": float(np.abs(a - b).max() / denom),
        "cosine": float((a.ravel() @ b.ravel()) /
                        (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
        "report": adapter.last_report,
    }
