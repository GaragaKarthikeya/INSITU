"""Run every check. `python -m kernel.tests.run [pattern]`"""

import sys

from .harness import run

MODULES = [
    "kernel.tests.test_numerics",
    "kernel.tests.test_codebook",
    "kernel.tests.test_rotation",
    "kernel.tests.test_quantize",
    "kernel.tests.test_attention",
    "kernel.tests.test_kernel",
    "kernel.tests.test_hardware",
    "kernel.tests.test_adapter",
]

if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(run([m for m in MODULES if pat in m]))
