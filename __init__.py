"""A compressed-KV attention kernel that never decompresses.

    from kernel import AttentionKernel, KernelConfig, ModelConfig

    cfg = KernelConfig(model=ModelConfig.tinyllama())
    k   = AttentionKernel(cfg)
    y, report = k.forward(x)              # (tokens, hidden) -> (tokens, hidden)

Layering, outermost first:

    kernel.py      composition. The only file that knows the pipeline order.
    weights.py     the four projections, with the offline folds applied.
    ops/           pure functions: project, rope, rotate, quantize, attention.
                   Values only; no op can see a hardware config.
    hw/            cycles and bytes, replayed from a trace. Never a value.
    cache/         KVCache interface; dense fp16 and compressed implementations.
    numerics/      fp.py (fp16 x fp16 -> fp32) and fixed.py (the ONLY narrowing).
    trace.py       the seam between the two halves.
    config.py      one validated dataclass tree.
"""

from .config import (ArrayConfig, AttentionConfig, CacheConfig, EncoderConfig,
                     FixedFormat, HardwareConfig, KernelConfig, MemoryConfig,
                     ModelConfig, QuantConfig, RotateConfig)
from .kernel import AttentionKernel, StepReport
from .trace import Op, Trace
from .weights import Weights

__all__ = [
    "AttentionKernel", "StepReport", "Weights", "Trace", "Op",
    "KernelConfig", "ModelConfig", "QuantConfig", "FixedFormat",
    "HardwareConfig", "ArrayConfig", "MemoryConfig",
    "RotateConfig", "EncoderConfig", "AttentionConfig", "CacheConfig",
]
