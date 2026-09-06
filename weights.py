"""Weights for one attention block, with the offline transforms already applied.

What "offline" means here
-------------------------
Two transforms belong to weight-loading, not to a decode step:

  * folding `R^-1` into `W_o`, so the block never leaves the rotated domain
  * casting to fp16, so a run never pays a conversion it could have paid once

Doing them in `Weights.prepare` rather than inside `forward` is what keeps the
claim "the inverse rotation costs nothing at runtime" true rather than
approximately true.

`Weights` is deliberately dumb about where the arrays came from. A checkpoint
loader, a random initialiser, and a torch state_dict all produce the same four
arrays, so there is one class and three constructors rather than three classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .numerics import fp
from .ops.output import fold_o_proj
from .ops.rotate import Rotation


@dataclass(frozen=True)
class Weights:
    """The four projections. `o` is already folded -- see `prepare`."""

    q: np.ndarray      # (num_heads * head_dim, hidden)
    k: np.ndarray      # (num_kv_heads * head_dim, hidden)
    v: np.ndarray
    o: np.ndarray      # (hidden, num_heads * head_dim), folded
    folded: bool = True

    # Per-head RMSNorm gains over head_dim, applied to q and k before RoPE.
    # Present in Qwen3, OLMo-2 and Gemma-3; absent in Llama and Mistral. None
    # means the architecture does not use QK-norm -- it is not a default that
    # can be quietly skipped, because a model that expects it produces
    # confidently wrong output without it.
    q_norm: np.ndarray | None = None
    k_norm: np.ndarray | None = None
    norm_eps: float = 1e-6

    # Projection biases. Qwen2.5 has them, Llama/Mistral/Qwen3 do not.
    q_bias: np.ndarray | None = None
    k_bias: np.ndarray | None = None
    v_bias: np.ndarray | None = None
    o_bias: np.ndarray | None = None

    @property
    def n_bytes(self) -> int:
        """fp16 bytes streamed to project one token. The dominant memory term."""
        return 2 * (self.q.size + self.k.size + self.v.size + self.o.size)

    @staticmethod
    def prepare(q, k, v, o, model: ModelConfig, rotation: Rotation,
                *, q_norm=None, k_norm=None, norm_eps: float = 1e-6,
                q_bias=None, k_bias=None, v_bias=None, o_bias=None) -> "Weights":
        """Validate shapes, fold `R` into `W_o`, cast everything to fp16.

        The output bias passes through the fold untouched: `y = acc @ W_o'.T +
        b_o`, and the fold only re-associates the matrix.
        """
        shapes = {
            "q": (model.q_out_features, model.hidden_size),
            "k": (model.kv_out_features, model.hidden_size),
            "v": (model.kv_out_features, model.hidden_size),
            "o": (model.hidden_size, model.q_out_features),
        }
        for name, arr, want in (("q", q, shapes["q"]), ("k", k, shapes["k"]),
                                ("v", v, shapes["v"]), ("o", o, shapes["o"])):
            if np.shape(arr) != want:
                raise ValueError(f"W_{name} is {np.shape(arr)}, expected {want}")

        for name, g in (("q_norm", q_norm), ("k_norm", k_norm)):
            if g is not None and np.shape(g) != (model.head_dim,):
                raise ValueError(
                    f"{name} is {np.shape(g)}, expected ({model.head_dim},): "
                    f"QK-norm is over head_dim, not hidden_size")
        if (q_norm is None) != (k_norm is None):
            raise ValueError("QK-norm must be given for both q and k, or neither")

        o_folded = fold_o_proj(np.asarray(o, dtype=np.float64), rotation, model)
        f32 = lambda a: None if a is None else np.asarray(a, dtype=np.float32)
        out = Weights(
            q=fp.to_fp16(q), k=fp.to_fp16(k), v=fp.to_fp16(v), o=fp.to_fp16(o_folded),
            q_norm=f32(q_norm), k_norm=f32(k_norm), norm_eps=norm_eps,
            q_bias=f32(q_bias), k_bias=f32(k_bias),
            v_bias=f32(v_bias), o_bias=f32(o_bias),
        )
        for name, arr in (("W_q", out.q), ("W_k", out.k), ("W_v", out.v), ("W_o'", out.o)):
            fp.check_fp16_range(np.asarray(arr, dtype=np.float32), name)
        return out

    @staticmethod
    def random(model: ModelConfig, rotation: Rotation, seed: int = 0,
               scale: float | None = None) -> "Weights":
        """Synthetic weights at the real dimensions.

        `scale` defaults to `1/sqrt(fan_in)`, which is what an initialiser and a
        trained checkpoint both roughly produce, so activations land in the same
        order of magnitude as the real thing. This is stimulus for exercising
        the datapath, not a model-quality claim -- nothing about perplexity can
        be argued from a run on these.
        """
        rng = np.random.default_rng(seed)
        s = scale if scale is not None else 1.0 / np.sqrt(model.hidden_size)
        def w(shape):
            return rng.standard_normal(shape).astype(np.float32) * s
        return Weights.prepare(
            w((model.q_out_features, model.hidden_size)),
            w((model.kv_out_features, model.hidden_size)),
            w((model.kv_out_features, model.hidden_size)),
            w((model.hidden_size, model.q_out_features)),
            model, rotation,
        )

    @staticmethod
    def from_torch(state_dict, prefix: str, model: ModelConfig,
                   rotation: Rotation) -> "Weights":
        """Pull `{prefix}.{q,k,v,o}_proj.weight` out of a checkpoint's state dict.

        Accepts torch tensors or anything with `.detach().cpu().numpy()`; a
        plain dict of numpy arrays works too, which is what keeps torch an
        optional dependency of this package rather than a required one.
        """
        def get(name):
            t = state_dict[f"{prefix}.{name}_proj.weight"]
            return t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t)
        return Weights.prepare(get("q"), get("k"), get("v"), get("o"), model, rotation)
