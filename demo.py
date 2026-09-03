"""What one attention block costs. `python -m kernel.demo [context]`

Runs a TinyLlama-shaped block on synthetic weights and reports where the time
and the bytes actually go. The weights are random -- this exercises the
datapath and the memory model, and says NOTHING about model quality.
"""

from __future__ import annotations

import sys

import numpy as np

from . import (AttentionKernel, HardwareConfig, KernelConfig, ModelConfig,
               QuantConfig, Trace, Weights)
from .hw import replay
from .ops.rotate import Rotation


def main(context: int = 512) -> None:
    m = ModelConfig.tinyllama()
    quant = QuantConfig(key_bits=3, value_bits=2)
    hw = HardwareConfig.sized_for(m, side=128)
    cfg = KernelConfig(model=m, quant=quant, hw=hw)
    print(cfg.describe(), "\n")

    weights = Weights.random(m, Rotation.from_seed(m.head_dim, quant.seed))
    rng = np.random.default_rng(0)

    for compressed in (True, False):
        k = AttentionKernel(cfg, weights, capacity=context + 8, compressed=compressed)
        k.forward(rng.standard_normal((context, m.hidden_size)).astype(np.float32) * 0.5)

        t = Trace()
        with t.phase("decode"):
            _, rep = k.forward(
                rng.standard_normal(m.hidden_size).astype(np.float32) * 0.5, trace=t)
        perf = replay(t, hw, n_tokens=1)

        label = "COMPRESSED" if compressed else "DENSE fp16"
        print(f"=== {label}, one decode step at context {rep.context} ===")
        print(rep.summary())
        print(perf.summary(1))

        print(f"ratio      weights are "
              f"{perf.weight_bytes / max(perf.cache_bytes, 1):.0f}x the KV traffic\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 512)
