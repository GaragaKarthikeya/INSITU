"""The randomized Walsh-Hadamard rotation.

    R = (1/sqrt(d)) . H_d . D     with D = diag(+-1) drawn from `seed`

Three properties, each of which the rest of the design leans on:

1. **Orthonormal.** So `<R q, R k> == <q, k>`: a query rotated once at the top
   of a decode step scores correctly against keys that were rotated when they
   were written, and no key ever has to be rotated back. That is what makes an
   O(context) inverse rotation collapse to a single O(1) one.

2. **Exact on integers.** `H` is adds and subtracts only, and `D` is a sign
   flip. The only truncation in the whole rotation is the final `1/sqrt(d)`
   scale, and for even powers of two that is an exact right shift. Nothing
   here is a source of divergence between two implementations.

3. **Gaussianising.** A rotation by a dense random orthogonal matrix makes each
   output channel a weighted sum of all `d` inputs, so it tends to Gaussian.
   `H D` is the cheap structured stand-in -- O(d log d) adds instead of O(d^2)
   multiplies -- and it is why `Codebook.gaussian` needs no calibration.

HOW MANY ROUNDS, AND A CORRECTION
---------------------------------
One round of `H D` does not land on a Gaussian; it OVERSHOOTS. Measured excess
kurtosis (`tests/test_rotation.py`):

    input                       1 round    2 rounds
    one-hot (pathological)        -2.00       -0.25
    4 outlier channels x30        -0.87       -0.14
    student-t, df=2.5             -0.26       -0.08

`H D` on a one-hot vector gives `+-1/sqrt(d)` in every channel -- a two-point
distribution, excess kurtosis exactly -2. A second round with an independent
sign diagonal fixes that, for one more `d log d` pass of adds.

**And it makes the model worse.** Measured end to end on WikiText-2 (12
windows, paired, TinyLlama at 1024 context, 4-bit keys / 2-bit values), one
round beats two by 1.3% perplexity, significant at 2.8 standard errors. Three
rounds is indistinguishable from two.

The lesson is about the proxy, not the transform. Kurtosis says how Gaussian
the channels look; it says nothing about the truncation each round costs. The
scale is applied INSIDE every round (see `apply`), so a second round floors the
value a second time -- and that appears to cost more than the distributional
gain returns. A statistic that looks like the objective is not the objective,
and the only way to find that out was to measure the thing being optimised.

So the default is ONE round. It is also half the butterfly.

The inverse is the transpose, and the design never computes it: `R^-1` is
folded into the output projection once, offline (`ops.output.fold_o_proj`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import INT, Q, fwht, inv_sqrt_q15, lshift, rshift, sqrt_shift
from ..trace import Op, Trace, record


@dataclass(frozen=True)
class Rotation:
    """`d`-dimensional randomized Hadamard rotation, fixed at construction.

    The sign diagonal is part of the *format*, not of a run: a cache written
    under one seed is unreadable under another. It therefore travels with the
    weights.
    """

    d: int
    signs: np.ndarray          # (rounds, d) int64, +-1 -- one diagonal per round
    seed: int

    @property
    def rounds(self) -> int:
        return self.signs.shape[0]

    @staticmethod
    def for_quant(quant, d: int) -> "Rotation":
        """THE constructor. Every call site must use this one.

        The rotation is decided by more than one field of `QuantConfig`, and it
        must be IDENTICAL everywhere it is built -- the quantizer that writes
        the cache, the kernel that rotates the query, and the offline fold that
        folds its inverse into `W_o`. Building it from a subset of the fields in
        one of those places puts the block's output in a different basis from
        the one `W_o'` was folded for.

        That is not hypothetical: `rot_rounds` was threaded into the quantizer
        and the kernel but not into the adapter's fold, and the result was a
        perplexity of 3532 against a baseline of 9.42 -- catastrophic, but only
        for the two configurations that did not happen to match the default. A
        single constructor makes forgetting a field impossible rather than
        merely unlikely.
        """
        return Rotation.from_seed(d, quant.seed, quant.rot_rounds)

    @staticmethod
    def from_seed(d: int, seed: int, rounds: int = 1) -> "Rotation":
        if d & (d - 1):
            raise ValueError(f"d={d} must be a power of two for a Hadamard butterfly")
        if rounds < 1:
            raise ValueError(f"rounds={rounds} must be at least 1")
        # PCG64 with an explicit seed: reproducible across numpy versions and
        # platforms. All diagonals come from one stream, so `seed` alone names
        # the whole transform and a cache carries one number, not `rounds`.
        rng = np.random.Generator(np.random.PCG64(seed))
        signs = rng.integers(0, 2, size=(rounds, d)).astype(INT) * 2 - 1
        return Rotation(d=d, signs=signs, seed=seed)

    # -- the scale ---------------------------------------------------------

    @property
    def shift(self) -> int | None:
        """`1/sqrt(d)` as an exact right shift, when d is an even power of two."""
        return sqrt_shift(self.d)

    @property
    def inv_sqrt_q15(self) -> int:
        return inv_sqrt_q15(self.d)

    # -- the transform -----------------------------------------------------

    def apply(self, x, trace: Trace | None = None, unit: str = "") -> np.ndarray:
        """Rotate along the last axis. Fixed-point in, fixed-point out, same Q.

        The scale is the only truncation. When `d` is an odd power of two it is
        a Q15 reciprocal multiply instead of a shift, which costs one multiply
        and one extra truncation -- stated here rather than hidden, because it
        means d=32 and d=64 are not equally exact.
        """
        a = np.asarray(x)
        if a.shape[-1] != self.d:
            raise ValueError(f"rotate expects last axis {self.d}, got {a.shape[-1]}")

        s = self.shift
        out = np.asarray(a, dtype=INT)
        for r in range(self.rounds):
            # The scale is applied INSIDE each round rather than once at the
            # end. Deferring it would let the intermediate grow by sqrt(d) per
            # round, which at two rounds and d=128 is 7 extra bits of headroom
            # the accumulator would have to carry for no benefit.
            out = fwht(out * self.signs[r])
            out = rshift(out, s) if s is not None else \
                  rshift(out * INT(self.inv_sqrt_q15), 15)

        record(trace, Op.ROTATE, unit, m=self.d, k=int(np.prod(a.shape[:-1])) or 1,
               n=self.rounds, log2d=self.d.bit_length() - 1, exact_scale=s is not None)
        return out

    def matrix(self) -> np.ndarray:
        """The dense `d x d` form. For folding into `W_o` and for tests only.

        Never used on a hot path -- materialising it defeats the point of a
        butterfly -- but the fold is an offline weight transform, where an
        O(d^2) matrix is the clearest way to say what is happening.

        Round `r` is applied first, so the composite is `R_last ... R_0` --
        the same order `apply` runs them in. Getting this backwards produces a
        matrix that is still orthonormal, which is exactly why it would not be
        caught by an orthonormality check.
        """
        h = np.array([[1.0]])
        while h.shape[0] < self.d:
            h = np.block([[h, h], [h, -h]])
        m = np.eye(self.d)
        for r in range(self.rounds):
            m = ((h * self.signs[r]) / np.sqrt(self.d)) @ m
        return m
