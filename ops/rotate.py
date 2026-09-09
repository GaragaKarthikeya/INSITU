"""The randomized Walsh-Hadamard rotation.

    R = (1/sqrt(d)) . H_d . D     with D = diag(+-1) drawn from `seed`

The rest of the design leans on three properties of it.

1. It is orthonormal, so `<R q, R k> == <q, k>`. A query rotated once at the
   top of a decode step scores correctly against keys that were rotated when
   they were written, and no key ever has to be rotated back. That is what
   turns an O(context) inverse rotation into a single O(1) one.

2. It is exact on integers. `H` is adds and subtracts, `D` is a sign flip. The
   only truncation in the whole rotation is the final `1/sqrt(d)` scale, and
   for even powers of two even that is an exact right shift. So there is
   nothing in here for two implementations to disagree about.

3. It makes the channels look Gaussian. Rotating by a dense random orthogonal
   matrix makes every output channel a weighted sum of all `d` inputs, which
   pushes it towards a normal distribution. `H D` is the cheap structured
   stand-in for that -- O(d log d) adds instead of O(d^2) multiplies -- and it
   is the reason `Codebook.gaussian` needs no calibration data.

How many rounds
---------------
One round of `H D` does not land on a Gaussian, it overshoots. Measured excess
kurtosis (`tests/test_rotation.py`):

    input                       1 round    2 rounds
    one-hot (pathological)        -2.00       -0.25
    4 outlier channels x30        -0.87       -0.14
    student-t, df=2.5             -0.26       -0.08

`H D` on a one-hot vector gives `+-1/sqrt(d)` in every channel, which is a
two-point distribution with excess kurtosis of exactly -2. A second round with
an independent sign diagonal fixes that, for one more `d log d` pass of adds.

It also makes the model worse. Measured end to end on WikiText-2 (12 windows,
paired, TinyLlama at 1024 context, 4-bit keys and 2-bit values), one round
beats two by 1.3% perplexity at 2.8 standard errors. Three rounds is
indistinguishable from two.

The lesson is about the proxy rather than the transform. Kurtosis tells you how
Gaussian the channels look. It tells you nothing about what each round costs in
truncation, and the scale is applied inside every round (see `apply`), so a
second round rounds the value off a second time. That appears to cost more than
the better distribution gives back. A statistic that resembles the objective is
not the objective, and the only way to find that out was to measure the thing
we actually cared about.

So the default is one round, which is also half the butterfly.

Zero rounds
-----------
`rounds = 0` is the identity, and exists only as an experimental control.
Every measurement above compares one rotation against another; none of them
asks whether rotating at all beats not rotating, which is the claim the scalar
codebook actually rests on. `experiments/ablate.py`, group "rotation", asks it.

It is deliberately not a special case anywhere. `signs` is `(0, d)`, so the
loops in `apply` and `inverse_apply` run zero times and return their input
unchanged, and `matrix()` returns the `np.eye(d)` it starts from -- which means
`ops.output.fold_o_proj` folds an identity and leaves `W_o` alone. All three
call sites named in `for_quant` degrade together, which is the only way this
could be measured without reproducing the basis-mismatch bug described there.

The one thing that is a special case is the trace: no rotation records no
`ROTATE` work, because `hw.attn_block.RotateUnit` charges `max(n, 1)` rounds
per vector and would otherwise bill a round that never ran.

The inverse is just the transpose, and the design never computes it: `R^-1` is
folded into the output projection once, offline (`ops.output.fold_o_proj`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerics.fixed import INT, fwht, inv_sqrt_q15, rshift, sqrt_shift
from ..trace import Op, Trace, record

# The 1/sqrt(d) scale floors. Rounding to nearest was tried and dropped.
#
# Flooring is wrong in the same direction every time, so the d channel errors
# do not cancel the way rounding-to-nearest errors would. That is a real defect
# and it is visible at score level: holding the codes fixed and computing the
# reference in float64, rounding the query rotation to nearest cuts the part of
# the error that survives the softmax from 2.09 to 1.21 LSB of Q16.
#
# It does not reach perplexity. Measured on Llama 3.2 1B, WikiText-2, 2,048
# tokens, architecture B, floor -> nearest:
#
#     4b/2b   8.7209 -> 8.5495   -1.97%
#     4b/4b   7.5510 -> 7.5716   +0.27%
#     6b/2b   7.9541 -> 7.9595   +0.07%
#
# One gain and two results inside the ~1.4% this corpus resolves. So the
# adder buys nothing reliable, and it is not worth desyncing this from
# `rtl/rot_fwht.sv:62`, which is `>>> SHIFT` and must keep agreeing.
#
# The lesson is the same one the "how many rounds" note above records: a
# statistic that resembles the objective is not the objective. Score-level
# error behaved exactly as predicted and perplexity did not follow.


@dataclass(frozen=True)
class Rotation:
    """A `d`-dimensional randomized Hadamard rotation, fixed at construction.

    The sign diagonal is part of the format, not of a particular run. A cache
    written under one seed cannot be read under another, so the seed travels
    with the weights.
    """

    d: int
    signs: np.ndarray          # (rounds, d) int64, +-1 -- one diagonal per round
    seed: int

    @property
    def rounds(self) -> int:
        return self.signs.shape[0]

    @staticmethod
    def for_quant(quant, d: int) -> "Rotation":
        """The constructor. Every call site has to use this one.

        More than one field of `QuantConfig` decides the rotation, and it has to
        come out identical in all three places that build it: the quantizer that
        writes the cache, the kernel that rotates the query, and the offline fold
        that puts its inverse into `W_o`. Build it from a subset of the fields in
        any one of those and the block's output ends up in a different basis from
        the one `W_o'` was folded for.

        This has actually happened. `rot_rounds` was threaded into the quantizer
        and the kernel but not into the adapter's fold, and perplexity came out
        at 3532 against a baseline of 9.42 -- but only for the two configurations
        that did not happen to match the default, so it looked fine most of the
        time. Having one constructor makes forgetting a field impossible instead
        of merely unlikely.
        """
        return Rotation.from_seed(d, quant.seed, quant.rot_rounds)

    @staticmethod
    def from_seed(d: int, seed: int, rounds: int = 1) -> "Rotation":
        if d & (d - 1):
            raise ValueError(f"d={d} must be a power of two for a Hadamard butterfly")
        if rounds < 0:
            raise ValueError(f"rounds={rounds} must not be negative")
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

        The scale is the only truncation in here. When `d` is an odd power of
        two the scale becomes a Q15 reciprocal multiply rather than a shift,
        which costs one multiply and one more truncation. Worth saying out loud:
        it means d=32 and d=64 are not equally exact.
        """
        a = np.asarray(x)
        if a.shape[-1] != self.d:
            raise ValueError(f"rotate expects last axis {self.d}, got {a.shape[-1]}")

        s = self.shift
        out = np.asarray(a, dtype=INT)
        for r in range(self.rounds):
            # The scale goes inside each round rather than once at the end.
            # Leaving it to the end would let the intermediate grow by sqrt(d)
            # per round, and at two rounds with d=128 that is 7 more bits of
            # headroom the accumulator would have to carry for nothing.
            out = fwht(out * self.signs[r])
            out = rshift(out, s) if s is not None else \
                  rshift(out * INT(self.inv_sqrt_q15), 15)

        if self.rounds:
            record(trace, Op.ROTATE, unit, m=self.d,
                   k=int(np.prod(a.shape[:-1])) or 1, n=self.rounds,
                   log2d=self.d.bit_length() - 1, exact_scale=s is not None)
        return out

    def inverse_apply(self, y, trace: Trace | None = None, unit: str = "") -> np.ndarray:
        """Undo `apply`. Fixed-point in, fixed-point out, same Q.

        `apply` computes `x @ R.T`, so this computes `y @ R`. The two are
        mirror images: `apply` flips signs, transforms, then scales; this
        transforms, flips signs, then scales, and runs the rounds in reverse.

        The design never calls this. `R^-1` is folded into `W_o` offline
        (`ops.output.fold_o_proj`), so nothing on the hot path rotates back.
        It exists for architecture A, which reconstructs into the original
        basis the way a compressed format's decode rule specifies, and so has
        to undo the rotation once per cached token.

        It is not an exact inverse in fixed point, and that is the cost being
        measured: each round ends in a truncation, so a round trip loses bits
        that staying in the rotated domain never touches.
        """
        a = np.asarray(y)
        if a.shape[-1] != self.d:
            raise ValueError(f"inverse expects last axis {self.d}, got {a.shape[-1]}")

        s = self.shift
        out = np.asarray(a, dtype=INT)
        for r in reversed(range(self.rounds)):
            out = fwht(out) * self.signs[r]
            out = rshift(out, s) if s is not None else \
                  rshift(out * INT(self.inv_sqrt_q15), 15)

        if self.rounds:
            record(trace, Op.ROTATE, unit, m=self.d,
                   k=int(np.prod(a.shape[:-1])) or 1, n=self.rounds,
                   log2d=self.d.bit_length() - 1, exact_scale=s is not None)
        return out

    def matrix(self) -> np.ndarray:
        """The dense `d x d` form, for folding into `W_o` and for tests.

        Never used on a hot path -- building it defeats the point of having a
        butterfly. But the fold is an offline weight transform, and there an
        O(d^2) matrix is the clearest way to write what is happening.

        Round `r` is applied first, so the composite is `R_last ... R_0`, which
        is the order `apply` runs them in. Get this backwards and you still get
        an orthonormal matrix, which is exactly why an orthonormality check
        would not catch it.
        """
        h = np.array([[1.0]])
        while h.shape[0] < self.d:
            h = np.block([[h, h], [h, -h]])
        m = np.eye(self.d)
        for r in range(self.rounds):
            m = ((h * self.signs[r]) / np.sqrt(self.d)) @ m
        return m
