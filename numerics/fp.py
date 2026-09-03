"""Floating-point arithmetic as the systolic arrays actually perform it.

The arrays multiply in fp16 and accumulate in fp32. That is what tensor
hardware does, and it is not a detail: a pure-fp16 accumulation over a
2048-long reduction loses several bits to swamping, and a pure-fp32 one is not
what the hardware would be built as. Modelling either would give a number the
silicon does not produce.

Everything here is a pure function over numpy arrays. No state, no config.
"""

from __future__ import annotations

import numpy as np

FP16 = np.float16
ACC = np.float32


def to_fp16(x: np.ndarray) -> np.ndarray:
    """Round to nearest even, IEEE754 binary16. Overflow saturates to inf."""
    return np.asarray(x, dtype=FP16)


def matvec(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`w @ x` with fp16 inputs, fp32 products and fp32 accumulation.

    `w` is (out, in) and `x` is (in,) or (batch, in) -- the projection weight
    layout every checkpoint uses, kept rather than transposed so a weight
    tensor can be handed over without a copy.

    The accumulation ORDER is left to numpy here. Order matters in floating
    point, and the cycle-accurate array in `hw.pe_array` reduces in its own
    order; `tests/test_pe_array.py` pins the two together rather than assuming
    they agree.
    """
    w16 = to_fp16(w)
    x16 = to_fp16(x)
    return (w16.astype(ACC) @ x16.astype(ACC).T).T.astype(ACC)


def matvec_ordered(w: np.ndarray, x: np.ndarray, chunk: int) -> np.ndarray:
    """`matvec`, reducing in explicit `chunk`-sized partial sums.

    This is the accumulation tree an array of `chunk` rows produces: each pass
    over the reduction dimension yields a partial sum, and the partials are
    added in pass order. Given the same `chunk` as the array's row count, the
    result is bit-identical to the PE model, which is what makes the PE model
    checkable against something fast.
    """
    w16 = to_fp16(w).astype(ACC)
    x16 = to_fp16(x).astype(ACC)
    n_in = w16.shape[1]
    acc = np.zeros(w16.shape[0], dtype=ACC)
    for start in range(0, n_in, chunk):
        stop = min(start + chunk, n_in)
        acc = acc + w16[:, start:stop] @ x16[start:stop]
    return acc


def max_abs(x: np.ndarray) -> float:
    """Largest magnitude, ignoring non-finite entries.

    Used to report how close a projection came to fp16's 65504 ceiling. A
    silent inf in a weight stream looks exactly like a bad quantizer three
    stages later, so the kernel checks this rather than discovering it.
    """
    finite = np.isfinite(x)
    return float(np.abs(x[finite]).max()) if finite.any() else float("inf")


FP16_MAX = 65504.0


def check_fp16_range(x: np.ndarray, what: str) -> None:
    """Raise if `x` cannot survive an fp16 store."""
    if not np.all(np.isfinite(np.asarray(x, dtype=ACC))):
        raise ValueError(f"{what}: non-finite value before fp16 conversion")
    m = max_abs(np.asarray(x))
    if m > FP16_MAX:
        raise ValueError(
            f"{what}: max |x| = {m:.1f} exceeds fp16's {FP16_MAX:.0f} and would "
            f"become inf. This is a scaling problem in the weights, not a "
            f"rounding one, and clamping it here would hide it."
        )
