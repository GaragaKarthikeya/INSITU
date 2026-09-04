"""Attention that never decompresses.

THE IDEA
--------
A cached key is stored as `norm * centroid[code]` in the rotated domain. The
conventional thing to do is rebuild that dense vector and then dot it with the
query. This does not:

    <q, k>  =  <R q, R k>                      R is orthonormal
            =  <rq, norm * centroid[code]>
            =  norm * sum_j rq[j] * centroid[code[j]]

so the dot product runs directly on the codes. The query is rotated ONCE per
step; no cached token is ever touched by a rotation, and no dense key or value
is ever materialised. That is the whole architecture in three lines.

The same holds on the value side, which is why the output stays in the rotated
domain and `R^-1` is folded into `W_o` offline instead of being applied per
token.

TWO SOFTMAXES, AND WHY BOTH ARE HERE
------------------------------------
`attend_online` is what hardware runs: one pass over the cache, a running
maximum, and a rescale of the accumulator whenever the maximum moves. It reads
the cache once, which is the only thing that matters when the cache is off-die.

`attend_two_pass` computes every score first, takes the global maximum, and
then accumulates. Same arithmetic, but no rescale and therefore no rescale
truncation -- and it vectorises, so it is orders of magnitude faster in Python.

They are NOT bit-identical, and the difference is not a bug: the online form
truncates the accumulator once per rescale event. Keeping both makes that
difference a measured quantity (`AttentionStats.rescale_events`, and
`tests/test_softmax_equivalence.py`) instead of a claim in a comment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import FixedFormat, QuantConfig
from ..numerics.fixed import INT, Q, absmax, rshift
from ..trace import Op, Trace, record
from .quantize import CompressedKV, KVQuantizer

FLOAT64_EXACT = 1 << 53


def exact_int_matmul(a, b, b_absmax: int | None = None) -> np.ndarray:
    """Integer matrix product evaluated in float64 BLAS, exactly.

    numpy has no BLAS path for int64: a 512-cube int64 matmul runs at 1.1
    GFLOP/s on this machine against 474 GFLOP/s for float64. That is a factor
    of 427, and it is the difference between a 4096-context sweep taking hours
    and taking days.

    float64 represents every integer below 2^53 exactly, and addition and
    multiplication of exactly-represented integers whose result is also below
    2^53 are exact. So the product is not "accurate enough" -- it is the same
    integer, bit for bit.

    The bound is CHECKED, not assumed. `max|a| * max|b| * k` is the worst case
    (every term at maximum magnitude and the same sign); exceeding it raises
    rather than silently returning a rounded answer, which is the one failure
    mode of this trick and would be invisible downstream.
    """
    a = np.asarray(a, dtype=INT)
    b = np.asarray(b, dtype=INT)
    k = a.shape[-1]
    bound = absmax(a) * (b_absmax if b_absmax is not None else absmax(b)) * k
    if bound >= FLOAT64_EXACT:
        raise OverflowError(
            f"exact_int_matmul: worst-case magnitude {bound} >= 2^53, so float64 "
            f"would round. Reduce the reduction length ({k}) or narrow the "
            f"operands; do NOT relax this check."
        )
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(INT)


# --------------------------------------------------------------------------
# exp(-x) without a divider or a series
# --------------------------------------------------------------------------

class ExpLut:
    """`exp(-x)` as `2^-i * 2^-f`: a table on the fraction, a shift on the integer.

    `x` arrives in Q(acc_frac) and is always non-negative -- every call site
    passes a distance below a running maximum. Converting to base 2 turns the
    integer part into a shift, so the table only has to cover `[0, 1)` and is
    `2**lut_bits` entries regardless of how large `x` gets.

    A series expansion would need several multiplies and would still be wrong
    in the tail; a table over the whole range would be enormous. This is the
    standard split, and it is exact to within one LSB of Q(prob_frac).
    """

    def __init__(self, fmt: FixedFormat) -> None:
        self.fmt = fmt
        n = 1 << fmt.exp_lut_bits
        frac = np.arange(n, dtype=np.float64) / n
        self.table = np.rint(2.0 ** (-frac) * (1 << fmt.prob_frac)).astype(INT)
        # log2(e) in Q(acc_frac), used to convert exp(-x) into 2^(-x*log2e).
        self.log2e = INT(round(np.log2(np.e) * (1 << fmt.acc_frac)))
        # Beyond this the result is zero at Q(prob_frac) and the shift would
        # be undefined; clamping here is exact, not an approximation.
        self.max_int_part = fmt.prob_frac + 1

        # ONE FLAT TABLE, NOT A TABLE PLUS A SHIFT.
        #
        # The result is `table[addr] >> i` where `addr` is the low
        # `exp_lut_bits` of `t >> (acc_frac - exp_lut_bits)` and `i` is the
        # rest. So it is a function of that quantity ALONE, and the whole thing
        # collapses to a single gather over `(max_int_part + 1) << exp_lut_bits`
        # entries -- 4,608 int64s, 37 KB.
        #
        # This is not an approximation: every entry is computed by the exact
        # expression above, and `check_exp_lut_flat_matches_two_step` asserts
        # the two agree on every reachable input. It is worth doing because the
        # two-step form was 28% of total runtime -- eight passes over a
        # 17-million-element array where one gather suffices.
        self._shift = fmt.acc_frac - fmt.exp_lut_bits
        span = (self.max_int_part + 1) << fmt.exp_lut_bits
        self._span = span

        # THE INPUT IS CLAMPED, AND IT HAS TO BE.
        #
        # `delta * log2e` overflows int64 once delta exceeds about 2^47, which
        # a caller reaches trivially: masking a score to the format's floor and
        # subtracting it from a running maximum produces exactly that. The
        # old two-step form wrapped silently there and got away with it because
        # every caller discarded the masked lanes afterwards -- a wrong answer
        # that happened never to be read. The flat gather turns the same input
        # into an out-of-bounds index, which is how it was found.
        #
        # Clamping is EXACT, not a safety net: every delta at or above this
        # bound maps to the table's terminal zero entry, so the clamp cannot
        # change a result. It also lets callers stop masking the output.
        self._delta_max = INT(-(-((span << self._shift) << fmt.acc_frac)
                                // int(self.log2e)))
        u = np.arange(span + 1, dtype=INT)
        flat = self.table[u[:span] & (n - 1)] >> np.minimum(
            u[:span] >> fmt.exp_lut_bits, self.max_int_part)
        flat[(u[:span] >> fmt.exp_lut_bits) > self.max_int_part] = 0
        self._flat = np.concatenate([flat, np.zeros(1, dtype=INT)])

    def __call__(self, delta) -> np.ndarray:
        """`round(2^prob_frac * exp(-delta / 2^acc_frac))` for delta >= 0."""
        d = np.minimum(np.asarray(delta, dtype=INT), self._delta_max)
        if np.any(d < 0):
            raise ValueError("ExpLut expects a non-negative distance below the running max")
        t = rshift(d * self.log2e, self.fmt.acc_frac)          # x*log2(e), Q(acc_frac)
        return self._flat[np.minimum(t >> self._shift, self._span)]

    def two_step(self, delta) -> np.ndarray:
        """The unflattened form. The reference `__call__` is checked against."""
        d = np.minimum(np.asarray(delta, dtype=INT), self._delta_max)
        t = rshift(d * self.log2e, self.fmt.acc_frac)
        i = rshift(t, self.fmt.acc_frac)
        f = t & ((INT(1) << self.fmt.acc_frac) - 1)
        addr = rshift(f, self.fmt.acc_frac - self.fmt.exp_lut_bits)
        p = self.table[addr] >> np.minimum(i, self.max_int_part)
        return np.where(i > self.max_int_part, INT(0), p)


def reciprocal(l, fmt: FixedFormat) -> np.ndarray:
    """`1/l` in Q(recip_frac), for `l` in Q(prob_frac). Integer divide, once per step."""
    a = np.asarray(l, dtype=INT)
    num = INT(1) << (fmt.recip_frac + fmt.prob_frac)
    return np.where(a > 0, num // np.maximum(a, 1), INT(0))


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

@dataclass
class AttentionStats:
    """What the step did, collected as it ran rather than reconstructed after."""

    n_tokens: int = 0
    rescale_events: int = 0
    score_overflows: int = 0
    acc_overflows: int = 0
    max_abs_score: int = 0
    max_abs_acc: int = 0

    def merge(self, other: "AttentionStats") -> None:
        self.n_tokens = max(self.n_tokens, other.n_tokens)
        self.rescale_events += other.rescale_events
        self.score_overflows += other.score_overflows
        self.acc_overflows += other.acc_overflows
        self.max_abs_score = max(self.max_abs_score, other.max_abs_score)
        self.max_abs_acc = max(self.max_abs_acc, other.max_abs_acc)


# --------------------------------------------------------------------------
# the datapath
# --------------------------------------------------------------------------

class CompressedAttention:
    """Scores and accumulates directly on codes, for one KV head's cache.

    Stateless between calls. `q_rot` is expected to be rotated AND already
    scaled by `1/sqrt(head_dim)`: folding that scale into the query means the
    softmax needs no multiply of its own, and the query is one vector per step
    where the scores are one per cached token.
    """

    def __init__(self, quantizer: KVQuantizer, fmt: FixedFormat, quant: QuantConfig) -> None:
        self.z = quantizer
        self.fmt = fmt
        self.quant = quant
        self.exp = ExpLut(fmt)
        self.score_shift = fmt.score_shift_for(quant)
        self.acc_shift = fmt.acc_shift_for(quant)
        # 2^-acc_shift as an exact float64 power of two, so the float
        # accumulate's truncation is the integer shift and not an approximation.
        self._acc_scale = 2.0 ** -self.acc_shift
        self.score_q = Q(fmt.score_width, fmt.acc_frac, fmt.saturate)
        self.acc_q = Q(fmt.acc_width, fmt.acc_frac, fmt.saturate)

        # An OBSERVER on the online loop, for `hw/vectors.py`. Same contract as
        # `AttentionKernel.on_stage`: guarded by `is not None`, handed copies,
        # and read-only -- `softmax_online.sv` and `accum.sv` need the running
        # `m`, `l` and `acc` per token, and the alternative is a second
        # implementation of the recurrence in the vector generator, which is
        # exactly what these files exist to avoid.
        self.on_step = None

    # -- scoring -----------------------------------------------------------

    def scores(self, q_rot: np.ndarray, kv: CompressedKV,
               trace: Trace | None = None) -> tuple[np.ndarray, int]:
        """All scores for one query against `T` cached tokens, Q(acc_frac).

        `q_rot` is (..., d); `kv.k_idx` is (T, d). Returns (..., T).

        Planes are summed before the single shift. Shifting each term
        separately would truncate each one independently and can differ by a
        count from a datapath that shifts once, which is what hardware does.
        """
        centroids = self.z.key.codebook.centroids
        k_hat = centroids[kv.k_idx.astype(INT)]                 # (T, d) Q(centroid_frac)
        dot = np.einsum("...j,tj->...t", np.asarray(q_rot, dtype=INT), k_hat)
        raw = dot * kv.k_norm.astype(INT)                        # Q(qk+centroid+norm)
        s = rshift(raw, self.score_shift)
        overflow = self.score_q.overflow_count(s)
        s = self.score_q.clamp(s)
        record(trace, Op.SCORE, "attention", m=self.z.d, k=1,
               n=int(np.prod(np.shape(s))))
        return s, overflow

    def _accumulate_terms(self, p: np.ndarray, kv: CompressedKV) -> np.ndarray:
        """`(norm * prob * centroid) >> acc_shift`, one truncation per term.

        Each term is truncated BEFORE it joins the running sum, not after. That
        is what makes a vectorised sum equal to a sequential one: truncating
        the sum instead would depend on the order the terms arrived in.
        """
        centroids = self.z.value.codebook.centroids
        v_hat = centroids[kv.v_idx.astype(INT)]                  # (T, d)
        weight = kv.v_norm.astype(INT)[..., None] * p[..., None]  # (..., T, 1)
        return rshift(weight * v_hat, self.acc_shift)             # (..., T, d)

    # -- the two softmaxes -------------------------------------------------

    def attend_two_pass(self, q_rot: np.ndarray, kv: CompressedKV,
                        trace: Trace | None = None
                        ) -> tuple[np.ndarray, AttentionStats]:
        """Global maximum, then one accumulation. Vectorised; no rescale."""
        s, ovf = self.scores(q_rot, kv, trace)
        stats = AttentionStats(n_tokens=s.shape[-1], score_overflows=ovf,
                               max_abs_score=int(np.abs(s).max(initial=0)))

        m = s.max(axis=-1, keepdims=True)
        p = self.exp(m - s)                                       # Q(prob_frac)
        l = p.sum(axis=-1)                                        # Q(prob_frac)

        terms = self._accumulate_terms(p, kv)
        acc = terms.sum(axis=-2)
        stats.acc_overflows = self.acc_q.overflow_count(acc)
        acc = self.acc_q.clamp(acc)
        stats.max_abs_acc = int(np.abs(acc).max(initial=0))

        record(trace, Op.SOFTMAX, "attention", m=1, n=int(np.prod(l.shape)))
        record(trace, Op.ACCUMULATE, "attention", m=self.z.d, n=int(np.prod(p.shape)))
        return self._finalize(acc, l), stats

    def attend_online(self, q_rot: np.ndarray, kv: CompressedKV,
                      trace: Trace | None = None
                      ) -> tuple[np.ndarray, AttentionStats]:
        """One streaming pass, running maximum, rescale on a new maximum.

        This is the hardware's order and the hardware's truncation pattern. The
        cache is read exactly once, which is the only property that matters
        when it lives off-die.
        """
        s, ovf = self.scores(q_rot, kv, trace)
        s = np.atleast_2d(s)                                      # (H, T)
        n_heads, n_tok = s.shape
        stats = AttentionStats(n_tokens=n_tok, score_overflows=ovf,
                               max_abs_score=int(np.abs(s).max(initial=0)))

        d = self.z.d
        acc = np.zeros((n_heads, d), dtype=INT)
        l = np.zeros(n_heads, dtype=INT)
        m = np.full(n_heads, self.score_q.lo, dtype=INT)
        unity = INT(1) << self.fmt.prob_frac

        v_centroids = self.z.value.codebook.centroids
        for t in range(n_tok):
            st = s[:, t]
            grew = st > m
            if grew.any():
                # exp(-(new_max - old_max)) pulls everything already summed
                # down onto the new scale. Rare in practice, so hardware can
                # stall here rather than carry a permanent d-wide multiplier.
                delta = np.where(grew, st - m, 0)
                factor = np.where(grew, self.exp(delta), unity)
                acc = rshift(acc * factor[:, None], self.fmt.prob_frac)
                l = rshift(l * factor, self.fmt.prob_frac)
                m = np.where(grew, st, m)
                stats.rescale_events += int(grew.sum())
                record(trace, Op.RESCALE, "attention", m=d, n=int(grew.sum()))

            p = self.exp(m - st)
            l = l + p
            v_hat = v_centroids[kv.v_idx[t].astype(INT)]
            acc = acc + rshift(
                (kv.v_norm[t].astype(INT) * p)[:, None] * v_hat, self.acc_shift
            )
            stats.acc_overflows += self.acc_q.overflow_count(acc)
            acc = self.acc_q.clamp(acc)

            if self.on_step is not None:
                self.on_step(t, s=st, m=m, p=p, l=l, acc=acc,
                             grew=grew, factor=factor if grew.any() else None)

        stats.max_abs_acc = int(np.abs(acc).max(initial=0))
        record(trace, Op.SOFTMAX, "attention", m=1, n=n_heads * n_tok)
        record(trace, Op.ACCUMULATE, "attention", m=d, n=n_heads * n_tok)
        out = self._finalize(acc, l)
        return out.reshape(np.shape(q_rot)[:-1] + (d,)), stats

    # -- the batched causal path -------------------------------------------

    def attend_causal_batch(self, q_rot: np.ndarray, kv: CompressedKV,
                            base: int = 0, tile: int = 256,
                            trace: Trace | None = None
                            ) -> tuple[np.ndarray, AttentionStats]:
        """Every query token of a prefill against one KV head, in one pass.

        `q_rot` is (n_q, heads, d); query token `i` attends cache tokens
        `[0, base + i]`. Returns (n_q, heads, d).

        THIS DOES NOT CHANGE THE MODELLED HARDWARE. It is an evaluation
        strategy, not a datapath: the caller still emits one MEM_READ per query
        token, so the trace -- and therefore every byte, cycle and energy
        figure -- is identical to the token-at-a-time path. That separation is
        exactly what `trace.py` exists to provide, and
        `tests/test_attention.py::check_causal_batch_is_bit_identical` pins the
        values while `test_kernel.py::check_batching_does_not_change_the_trace`
        pins the accounting.

        `tile` bounds peak memory only. It is NOT a query-tile in the hardware
        sense and must not be reported as one.

        The accumulate cannot be a plain matmul, because each term is truncated
        BEFORE it joins the sum and that is load-bearing. It is instead
        decomposed over the `2**value_bits` distinct centroids: within one code
        value every term shares a multiplier, so the truncation happens first
        and what remains is a 0/1 matrix product. Exact, and `2**value_bits`
        BLAS calls instead of one enormous elementwise temporary.
        """
        q = np.asarray(q_rot, dtype=INT)
        if q.ndim == 2:
            q = q[:, None, :]
        n_q, heads, d = q.shape
        n_tok = kv.k_idx.shape[0]

        k_hat = self.z.key.codebook.centroids[kv.k_idx.astype(INT)]   # (T, d)
        k_hat_max = absmax(k_hat)
        k_norm = kv.k_norm.astype(INT)
        v_norm_f = kv.v_norm.astype(np.float64)
        v_cent = self.z.value.codebook.centroids
        v_cent_max = absmax(v_cent)
        # Everything the exactness guards need, computed once for the whole
        # batch instead of once per tile per code value. A guard that walks the
        # array it is guarding costs more than the work it protects.
        score_scale = 2.0 ** -self.score_shift
        k_norm_f = kv.k_norm.astype(np.float64)
        k_norm_max = absmax(kv.k_norm)
        # 0/1 selector per value code, built once for the whole batch.
        selector = [(kv.v_idx == v).astype(np.float64)
                    for v in range(self.z.value.codebook.size)]

        out = np.empty((n_q, heads, d), dtype=INT)
        stats = AttentionStats(n_tokens=n_tok)
        positions = np.arange(n_tok)

        for lo in range(0, n_q, tile):
            hi = min(lo + tile, n_q)
            nt = hi - lo

            qt = q[lo:hi].reshape(-1, d)
            q_max = absmax(qt)
            if q_max * k_hat_max * d >= FLOAT64_EXACT:
                raise OverflowError(
                    f"score: |q|*|k_hat|*d = {q_max * k_hat_max * d} >= 2^53")
            dot_f = qt.astype(np.float64) @ k_hat.T.astype(np.float64)

            # The score truncation also runs in float64 when the operands prove
            # it is exact, which is the ordinary case: `dot * k_norm` is an
            # exact float64 integer below 2^53, and the shift is a power-of-two
            # scale. The int64 fallback is kept because the bound depends on
            # the data, not only on the format, so it can genuinely be reached.
            if q_max * k_hat_max * d * k_norm_max < FLOAT64_EXACT:
                s = np.floor((dot_f * k_norm_f) * score_scale).astype(INT)
            else:
                s = rshift(dot_f.astype(INT) * k_norm, self.score_shift)
            s, ovf, peak = self.score_q.check_and_clamp(s)
            stats.score_overflows += ovf
            stats.max_abs_score = max(stats.max_abs_score, peak)
            s = s.reshape(nt, heads, n_tok)

            # Causal mask. Invalid entries are pinned to the format's floor so
            # the running maximum is taken over the prefix only; their
            # probability is then exactly zero and contributes nothing.
            valid = (positions[None, :] <= (base + np.arange(lo, hi))[:, None])
            s = np.where(valid[:, None, :], s, self.score_q.lo)

            # Masked lanes are pinned far enough below the running maximum
            # that `ExpLut`'s clamp maps them to exactly zero, so they need no
            # second masking pass -- and, more importantly, no lane ever
            # reaches the exp with a delta that would overflow.
            m = s.max(axis=-1, keepdims=True)
            p = self.exp(m - s)
            l = p.sum(axis=-1)

            # The accumulate runs in float64 from here, and it is EXACT.
            #
            # `rshift(x, s)` is `floor(x / 2^s)`. In float64, `w * c` is an
            # exact integer while it stays under 2^53, and scaling by 2^-s is
            # exact because the scale is a power of two, so `floor((w*c) *
            # 2^-s)` is the same integer as the int64 shift -- and it feeds the
            # matmul without a conversion. The int64 path cost a multiply, a
            # shift and an astype over a 17-million-element array per code
            # value; this costs a multiply and a floor.
            #
            # The bound is checked, not assumed, exactly as in
            # `exact_int_matmul`.
            w = p.reshape(-1, n_tok).astype(np.float64) * v_norm_f
            w_max = absmax(p) * absmax(kv.v_norm)
            if w_max * v_cent_max >= FLOAT64_EXACT:
                raise OverflowError(
                    f"accumulate: |w|*|c| = {w_max * v_cent_max} >= 2^53, so the "
                    f"float64 truncation would not match the integer shift"
                )
            # |term| <= |w|*|c|*2^-shift, bounded analytically rather than by
            # walking the array once per code value.
            if w_max * v_cent_max * self._acc_scale * n_tok >= FLOAT64_EXACT:
                raise OverflowError(
                    f"accumulate: the reduction over {n_tok} tokens exceeds 2^53; "
                    f"lower `tile` will not help"
                )
            # Scale once, not once per code. `w * 2^-acc_shift` is exact
            # (a power-of-two scale never rounds), so `floor(w_s * c)` is the
            # same integer as `floor((w * c) * 2^-acc_shift)` -- but it is one
            # multiply per code instead of two, and building these terms costs
            # four times what the matmuls that consume them cost.
            w_s = w * self._acc_scale
            acc = np.zeros((nt * heads, d), dtype=np.float64)
            for v, sel in enumerate(selector):
                acc += np.floor(w_s * float(v_cent[v])) @ sel
            acc, ovf, peak = self.acc_q.check_and_clamp(acc.astype(INT))
            stats.acc_overflows += ovf
            stats.max_abs_acc = max(stats.max_abs_acc, peak)
            acc = acc.reshape(nt, heads, d)
            out[lo:hi] = self._finalize(acc, l)

        # The CAUSAL counts, not the rectangular ones. Query token i scores
        # against base+i+1 cached tokens, not against all n_tok -- the batch
        # computes a full rectangle and masks, but the modelled design does
        # only the triangle, and it is the design the trace describes.
        causal = heads * sum(base + i + 1 for i in range(n_q))
        record(trace, Op.SCORE, "attention", m=d, n=causal)
        record(trace, Op.SOFTMAX, "attention", m=1, n=n_q * heads)
        record(trace, Op.ACCUMULATE, "attention", m=d, n=causal)
        return out.reshape(np.shape(q_rot)), stats

    def attend_online_batch(self, q_rot: np.ndarray, kv: CompressedKV,
                            base: int = 0, trace: Trace | None = None
                            ) -> tuple[np.ndarray, AttentionStats]:
        """The ONLINE softmax over a whole prefill, one cached token per step.

        This is what the hardware does: a single pass over the cache carrying a
        running maximum, rescaling the accumulator whenever that maximum moves.
        `attend_online` already expresses it, but one query at a time, which is
        O(n^2) Python iterations for a prefill and hours per window.

        Here the loop is over CACHED TOKENS and every query is advanced
        together. The step is still exactly one cached token, so the sequence of
        rescale events -- and therefore the truncation pattern, which is the
        whole reason the online and two-pass forms differ -- is identical to the
        per-query loop. `check_online_batch_is_bit_identical` pins that.

        Causality is handled by only admitting query `q` once the loop reaches
        token `q`: before that the token is in its future and must not touch its
        running maximum.
        """
        q = np.asarray(q_rot, dtype=INT)
        if q.ndim == 2:
            q = q[:, None, :]
        n_q, heads, d = q.shape
        n_tok = kv.k_idx.shape[0]

        k_hat = self.z.key.codebook.centroids[kv.k_idx.astype(INT)]
        dot = exact_int_matmul(q.reshape(-1, d), k_hat.T)
        s = rshift(dot * kv.k_norm.astype(INT), self.score_shift)
        ovf = self.score_q.overflow_count(s)
        s = self.score_q.clamp(s).reshape(n_q, heads, n_tok)
        stats = AttentionStats(n_tokens=n_tok, score_overflows=ovf,
                               max_abs_score=absmax(s))

        v_cent = self.z.value.codebook.centroids
        v_hat = v_cent[kv.v_idx.astype(INT)]                    # (T, d)
        v_norm = kv.v_norm.astype(INT)
        unity = INT(1) << self.fmt.prob_frac

        acc = np.zeros((n_q, heads, d), dtype=INT)
        l = np.zeros((n_q, heads), dtype=INT)
        m = np.full((n_q, heads), self.score_q.lo, dtype=INT)
        q_index = base + np.arange(n_q)

        for t in range(n_tok):
            live = q_index >= t                    # queries this token is causal for
            if not live.any():
                break
            st = np.where(live[:, None], s[:, :, t], m)

            grew = st > m
            if grew.any():
                factor = np.where(grew, self.exp(np.maximum(st - m, 0)), unity)
                acc = rshift(acc * factor[:, :, None], self.fmt.prob_frac)
                l = rshift(l * factor, self.fmt.prob_frac)
                m = np.where(grew, st, m)
                stats.rescale_events += int(grew.sum())

            p = np.where(live[:, None], self.exp(m - st), INT(0))
            l = l + p
            acc = acc + rshift((v_norm[t] * p)[:, :, None] * v_hat[t], self.acc_shift)
            stats.acc_overflows += self.acc_q.overflow_count(acc)
            acc = self.acc_q.clamp(acc)

        stats.max_abs_acc = absmax(acc)
        causal = heads * sum(base + i + 1 for i in range(n_q))
        record(trace, Op.SCORE, "attention", m=d, n=causal)
        record(trace, Op.SOFTMAX, "attention", m=1, n=n_q * heads)
        record(trace, Op.ACCUMULATE, "attention", m=d, n=causal)
        record(trace, Op.RESCALE, "attention", m=d, n=stats.rescale_events)
        return self._finalize(acc, l).reshape(np.shape(q_rot)), stats

    def _finalize(self, acc: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Divide by the softmax denominator. One reciprocal per step, not per token."""
        recip = reciprocal(l, self.fmt)[..., None]
        return rshift(acc * recip, self.fmt.recip_frac)
