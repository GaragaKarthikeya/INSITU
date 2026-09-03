"""The torch adapter's plumbing, against a stub decoder layer.

These run against a stub, so they stay fast and stay green without a 2 GB
download. They prove shapes, dtypes, devices and the batch guard.

They do NOT prove the signature matches an installed transformers -- the stub
cannot, by construction. That is `experiments/single_layer.py`'s job, and it
found two things this file did not: the real layer passes `position_ids` rather
than `cache_position`, and a grafted layer must advance `past_key_values` or
the model's own length bookkeeping goes wrong once every layer is replaced.
"""

import numpy as np

from kernel.config import ModelConfig, QuantConfig
from .harness import raises

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class _StubConfig:
    hidden_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 32
    rope_theta = 10000.0
    max_position_embeddings = 256


class _StubAttention:
    """Just enough of a Llama attention module to be read for its weights."""

    def __init__(self):
        import torch.nn as nn
        c = _StubConfig
        self.q_proj = nn.Linear(c.hidden_size, c.num_attention_heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.hidden_size, c.num_key_value_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.num_attention_heads * c.head_dim, c.hidden_size, bias=False)


def check_model_config_from_hf():
    m = ModelConfig.from_hf(_StubConfig)
    assert (m.hidden_size, m.num_heads, m.num_kv_heads, m.head_dim) == (128, 4, 2, 32)
    assert m.kv_groups == 2


def check_adapter_shapes_and_dtype():
    if not HAVE_TORCH:
        return
    from kernel.adapters.torch_llama import build_from_layer
    a = build_from_layer(_StubAttention(), _StubConfig, capacity=64)
    x = torch.randn(1, 5, 128)
    y, weights = a(x)
    assert y.shape == x.shape and y.dtype == x.dtype and weights is None


def check_adapter_accepts_two_dim_input():
    if not HAVE_TORCH:
        return
    from kernel.adapters.torch_llama import build_from_layer
    a = build_from_layer(_StubAttention(), _StubConfig, capacity=64)
    y, _ = a(torch.randn(3, 128))
    assert y.shape == (3, 128)


def check_adapter_refuses_a_padded_batch():
    """Silently attending across sequences is the failure this prevents."""
    if not HAVE_TORCH:
        return
    from kernel.adapters.torch_llama import build_from_layer
    a = build_from_layer(_StubAttention(), _StubConfig, capacity=64)
    raises(NotImplementedError, lambda: a(torch.randn(2, 5, 128)), "batch > 1")


def check_adapter_is_an_nn_module():
    """It must survive `layer.self_attn = adapter` without breaking the module tree."""
    if not HAVE_TORCH:
        return
    import torch.nn as nn
    from kernel.adapters.torch_llama import build_from_layer
    a = build_from_layer(_StubAttention(), _StubConfig, capacity=64)
    assert isinstance(a, nn.Module)


def check_adapter_carries_a_report():
    if not HAVE_TORCH:
        return
    from kernel.adapters.torch_llama import build_from_layer
    a = build_from_layer(_StubAttention(), _StubConfig, capacity=64)
    a(torch.randn(1, 4, 128))
    assert a.last_report is not None and a.last_report.context == 4
    a.reset()
    a(torch.randn(1, 2, 128))
    assert a.last_report.context == 2
