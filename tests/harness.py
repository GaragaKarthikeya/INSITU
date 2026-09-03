"""A test runner with no dependencies.

pytest is not available in this environment and the package has no reason to
require it: a test here is a module-level function named `check_*` that raises
on failure. `run.py` collects them.

Deliberately tiny. The value is in what the checks assert, not in the runner.
"""

from __future__ import annotations

import importlib
import time
import traceback


def approx(a, b, tol=1e-9, what="") -> None:
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{what}: shape {a.shape} != {b.shape}")
    err = np.abs(a - b).max() if a.size else 0.0
    if not err <= tol:
        raise AssertionError(f"{what}: max abs error {err:.3e} > {tol:.3e}")


def exact(a, b, what="") -> None:
    import numpy as np
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise AssertionError(f"{what}: not bit-identical")


def raises(exc, fn, what="") -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what}: expected {exc.__name__}, nothing raised")


def run(modules: list[str]) -> int:
    passed = failed = 0
    for name in modules:
        mod = importlib.import_module(name)
        checks = [(n, f) for n, f in vars(mod).items()
                  if n.startswith("check_") and callable(f)]
        for n, f in sorted(checks):
            t0 = time.time()
            try:
                f()
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}.{n}\n      {e}")
                if not isinstance(e, AssertionError):
                    traceback.print_exc()
            else:
                passed += 1
                print(f"ok    {name}.{n}  ({time.time() - t0:.2f}s)")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0
