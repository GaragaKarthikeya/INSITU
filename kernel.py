"""`AttentionKernel` -- one attention block, embedding in, embedding out.

    forward(hidden_states, positions) -> hidden_states

That signature is why the package is shaped the way it is. It is what a
transformer layer already calls, so the kernel drops into a real model without
the model needing to know anything about rotations, codes or systolic arrays.

The pipeline
------------
    x  -> W_q,W_k,W_v      three arrays, fp16 x fp16 -> fp32
       -> fp32 to Q(qk_frac)                     [ops.convert, the one cast]
       -> RoPE                                    [position enters here]
       -> R = randomized Hadamard                 [q, k and v, all three]
       -> quantize K,V, write to cache            [never rotated again]
       -> attend on codes                         [no dequantisation, ever]
       -> W_o'                                    [R^-1 already folded in]

Steps 3 to 6 are what the FPGA runs. See `host/fpga_layers.py`.

Prefill and decode are the same code
------------------------------------
`forward` takes `(tokens, hidden)`. One row is a decode step, many rows are a
prefill. There is no separate decode path that has to be kept in agreement with
the prefill path, which is how accelerator models that only ever modelled the
steady state tend to go wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cache import CompressedCache, DenseCache, KVCache
from .config import KernelConfig
from .numerics.fixed import Q
from .ops.attention import AttentionStats, CompressedAttention
from .ops.convert import ConversionStats, to_fixed
from .ops.norm import rms_norm
from .ops.project import merge_heads, project, split_heads
from .ops.rope import RoPE
from .ops.rotate import Rotation
from .trace import Op, Trace, record
from .weights import Weights


@dataclass
class StepReport:
    """Everything one `forward` did, for the caller to keep or discard."""

    n_tokens: int = 0
    context: int = 0
    attention: AttentionStats = field(default_factory=AttentionStats)
    conversion: ConversionStats = field(default_factory=ConversionStats)
    trace: Trace | None = None

    def summary(self) -> str:
        a, c = self.attention, self.conversion
        s = (f"{self.n_tokens} token(s), context {self.context}\n"
             f"  rescales {a.rescale_events}  score_ovf {a.score_overflows}  "
             f"acc_ovf {a.acc_overflows}\n"
             f"  fp->fixed saturation {c.saturation_rate:.2%} "
             f"(max |x| = {c.max_abs_float:.3f})")
        if self.trace is not None:
            s += "\n  " + self.trace.summary().replace("\n", "\n  ")
        return s


class AttentionKernel:
    """One attention block. Construct once per layer, call `forward` per step.

    `compressed=False` swaps in the fp16 baseline cache and skips quantisation
    entirely, leaving everything else -- projection, RoPE, rotation, softmax,
    fixed-point widths -- identical. That is what makes an A/B attributable to
    the compression rather than to two different implementations.
    """

    def __init__(self, config: KernelConfig, weights: Weights | None = None, *,
                 capacity: int = 2048, compressed: bool = True) -> None:
        self.cfg = config
        m = config.model
        self.rotation = Rotation.for_quant(config.quant, m.head_dim)
        self.rope = RoPE.build(m.head_dim, m.max_position, m.rope_theta,
                               config.fmt.centroid_frac, m.rope_scaling)
        self.qk_q = Q(config.fmt.qk_width, config.fmt.qk_frac, config.fmt.saturate)
        self.compressed = compressed

        self.weights = weights if weights is not None else \
            Weights.random(m, self.rotation, seed=0)

        if compressed:
            self.cache: KVCache = CompressedCache(
                m.num_kv_heads, m.head_dim, capacity, config.quant, config.fmt)
            self.attn = CompressedAttention(
                self.cache.quantizer, config.fmt, config.quant)
        else:
            self.cache = DenseCache(m.num_kv_heads, m.head_dim, capacity)
            self.attn = None

        # 1/sqrt(head_dim). Folded into the query in float, before the single
        # fp->fixed cast, so it costs no truncation of its own.
        self.attn_scale = 1.0 / np.sqrt(m.head_dim)

        # How many query tokens the batched prefill evaluates at once. This
        # bounds peak memory and nothing else -- it is invisible to the values
        # and to the trace. It is not a hardware query tile and must not be
        # reported as one.
        self.batch_tile = 256

        # Forces the token-at-a-time path even during prefill. For tests only:
        # it is the reference the batched path gets checked against, both in
        # values and in trace records. Do not set it on a measurement run -- it
        # is 20x slower and gives the identical answer.
        self.force_token_loop = False

        # "two_pass" takes a global maximum and then accumulates. "online"
        # carries a running maximum and rescales the accumulator whenever it
        # moves. The hardware runs online, because that reads the cache once,
        # and the online form truncates the accumulator at every rescale -- so
        # the two do not agree exactly. Selecting online forces the
        # token-at-a-time path and is much slower; it is here so the gap can be
        # measured rather than assumed to be small.
        self.softmax = "two_pass"

        # An observer, for the vector generator. Called as
        # `on_stage(name, array)` at each named point below. The array it is
        # handed is the one the datapath already computed, and it is never read
        # back. Leaving it at None costs one `is not None` check per stage,
        # which is the same guard `record` uses for the trace and for the same
        # reason: golden vectors have to come off the unmodified kernel, or they
        # are vectors for a different machine.
        self.on_stage = None

        # Stop at the seam. With this set, `forward` runs the host's half --
        # projections, QK-norm, the scale, the fp -> fixed cast, RoPE, the
        # rotation and the cache write -- taps every stage as usual, and then
        # returns, without scoring, softmaxing or accumulating anything.
        #
        # That is not an optimisation of the attention: it is the observation
        # that when the PL is doing the attention, the numpy attention is a
        # per-step check and nothing else. `collect(..., verify=False)` sets it
        # for the duration of one call. The return is `(None, report)` --
        # `None` rather than a zero array, so a caller that wanted the output
        # and did not want this fails immediately instead of consuming zeros.
        #
        # Never set on a run whose answer is used. It is off by default and
        # `forward`'s numbers are unchanged when it is.
        self.ingress_only = False

        # Where the float -> fixed cast sits relative to RoPE. See `forward`.
        #
        # True puts it at the seam, which is where the boundary actually is:
        # the host holds the model in float, so RoPE costs it nothing extra,
        # and the fixed-point world starts at the wire. Measured against the
        # other ordering with quantisation off, the two agree to 7.7e-5 in
        # perplexity -- Q8.16 has enough fractional bits that two multiplies
        # and an add per channel pair lose nothing worth having.
        #
        # False keeps RoPE inside the fixed-point datapath. The board's
        # recorded results were taken that way, so anything compared against
        # them must set it.
        self.cast_after_rope = True

    def _tap(self, name: str, value) -> None:
        if self.on_stage is not None:
            self.on_stage(name, value)

    # -- the reduction chunk is the only thing hardware config leaks ---------

    def _chunk(self, which: str) -> int | None:
        if self.cfg.hw is None:
            return None
        return getattr(self.cfg.hw, f"{which}_array").rows

    # -- forward ------------------------------------------------------------

    def forward(self, hidden_states: np.ndarray, positions: np.ndarray | None = None,
                *, trace: Trace | None = None) -> tuple[np.ndarray, StepReport]:
        """`(tokens, hidden)` in, `(tokens, hidden)` out. One row is a decode step."""
        m = self.cfg.model
        x = np.asarray(hidden_states, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != m.hidden_size:
            raise ValueError(f"hidden size {x.shape[-1]} != {m.hidden_size}")
        n = x.shape[0]

        if positions is None:
            positions = np.arange(self.cache.length, self.cache.length + n)
        positions = np.asarray(positions, dtype=np.int64)
        if positions.shape != (n,):
            raise ValueError(f"expected {n} positions, got {positions.shape}")

        report = StepReport(n_tokens=n, trace=trace)
        conv = report.conversion

        # 1. project ------------------------------------------------------
        w = self.weights
        q = project(w.q, x, unit="q_array", bias=w.q_bias,
                    chunk=self._chunk("q"), trace=trace)
        k = project(w.k, x, unit="k_array", bias=w.k_bias,
                    chunk=self._chunk("k"), trace=trace)
        v = project(w.v, x, unit="v_array", bias=w.v_bias,
                    chunk=self._chunk("v"), trace=trace)
        record(trace, Op.MEM_READ, "memory", n_bytes=w.n_bytes, what="weights")

        q = split_heads(q, m.num_heads, m.head_dim)
        k = split_heads(k, m.num_kv_heads, m.head_dim)
        v = split_heads(v, m.num_kv_heads, m.head_dim)

        # QK-norm, if the architecture has it: per-head RMSNorm over head_dim,
        # before RoPE and before the fp -> fixed cast. V is not normalised.
        if w.q_norm is not None:
            q = rms_norm(q, w.q_norm, w.norm_eps)
            k = rms_norm(k, w.k_norm, w.norm_eps)
        # The attention scale comes after the norm: RMSNorm is invariant to the
        # scale of its input, so folding it in earlier would be erased.
        q = q * self.attn_scale

        # 2 and 3. the float -> fixed boundary, and RoPE --------------------
        #
        # Their ORDER is a design decision, not an implementation detail.
        #
        # By default the cast comes first and RoPE runs in fixed point. That
        # follows the package's rule of one cast, so everything downstream of
        # it is integer -- but it makes RoPE part of the fixed-point datapath
        # and charges it a truncation it would not otherwise pay.
        #
        # `cast_after_rope` puts the cast at the seam instead, where the real
        # boundary is: RoPE runs in float on the host, and the fixed-point
        # world begins at the wire. That is what a deployment would do, since
        # the host already has the model in float.
        #
        # V is not rotated by RoPE either way; it carries no position.
        if self.cast_after_rope:
            q = self.rope.apply_float(q, positions[:, None], trace, "rope")
            k = self.rope.apply_float(k, positions[:, None], trace, "rope")
            qi = to_fixed(q, self.qk_q, conv)
            ki = to_fixed(k, self.qk_q, conv)
            vi = to_fixed(v, self.qk_q, conv)
        else:
            qi = to_fixed(q, self.qk_q, conv)
            ki = to_fixed(k, self.qk_q, conv)
            vi = to_fixed(v, self.qk_q, conv)
            qi = self.rope.apply(qi, positions[:, None], self.cfg.fmt.qk_frac,
                                 trace, "rope")
            ki = self.rope.apply(ki, positions[:, None], self.cfg.fmt.qk_frac,
                                 trace, "rope")
        # V carries no position. RoPE on the value path would have to be undone
        # by the output projection, and there is nothing there to undo it.

        # The seam. Everything below this line belongs to the PL, and these
        # three tensors are exactly what crosses the wire, in Q8.16 on 24-bit
        # lanes.
        self._tap("wire.q", qi)
        self._tap("wire.k", ki)
        self._tap("wire.v", vi)

        # 4. rotate ----------------------------------------------------------
        # K and V are always rotated: it is what makes a scalar codebook a good
        # quantizer, so it belongs to the write path in both architectures.
        #
        # The QUERY is a different matter. B scores in the rotated domain and
        # needs it. A reconstructs cached keys back into the original basis and
        # scores there, so for A the rotated query is computed and discarded --
        # and charging the trace for it would overstate A's cost, which is the
        # one thing a comparison must not do. So A does not rotate the query,
        # and pays instead on the read side: an inverse rotation per cached
        # token, which is the trade this comparison exists to measure.
        kr = self.rotation.apply(ki, trace, "rotate")
        vr = self.rotation.apply(vi, trace, "rotate")
        if self.compressed and self.attn.score_mode == "dense":
            qr = qi
        else:
            qr = self.rotation.apply(qi, trace, "rotate")
        self._tap("rot.q", qr)
        self._tap("rot.k", kr)
        self._tap("rot.v", vr)

        # 5. write ----------------------------------------------------------
        base = self.cache.length
        if self.compressed:
            self.cache.append_rotated(kr, vr, trace)
        else:
            self.cache.append(self.qk_q.to_float(kr), self.qk_q.to_float(vr),
                              positions, trace)

        if self.ingress_only:
            # Everything above this line is the host's half and it has all run;
            # everything below is what the PL owns. See `__init__`.
            report.context = base + n
            return None, report

        # 6. attend ---------------------------------------------------------
        # Query heads that share a KV head are attended together. Under
        # grouped-query attention that means `kv_groups` queries against one
        # cache, and the scoring path is already batched over the leading axis
        # (`test_attention.py::check_batched_queries_match_one_at_a_time` pins
        # the batched and one-at-a-time results as identical). It is the same
        # arithmetic and `kv_groups` times fewer passes over the cache, which is
        # also what the hardware does.
        out = np.zeros((n, m.num_heads, m.head_dim), dtype=np.int64)
        if not self.compressed and n > 1 and not self.force_token_loop:
            # The dense control gets the same treatment, for the same reason
            # and with the same guarantee about the trace. It matters most on a
            # model without grouped-query attention: Llama-2-7B has 32 KV heads,
            # so at a 4096 context the token-at-a-time path would run one query
            # head against one cache 131,072 times per layer.
            for kvh in range(m.num_kv_heads):
                heads = slice(kvh * m.kv_groups, (kvh + 1) * m.kv_groups)
                y, st = self._attend_dense_batch(qr[:, heads], kvh, base, trace, n)
                out[:, heads] = y
                report.attention.merge(st)
        elif self.compressed and n > 1 and not self.force_token_loop:
            # Prefill: all query tokens against one pass over the cache, in
            # float64 BLAS. Bit-identical to the loop below, pinned by
            # test_attention.py::check_causal_batch_is_bit_identical, and the
            # trace matches too because `view` is told how many query tokens
            # this call serves. Neither the values nor the modelled hardware
            # change -- only how numpy is asked to evaluate them.
            for kvh in range(m.num_kv_heads):
                heads = slice(kvh * m.kv_groups, (kvh + 1) * m.kv_groups)
                kv = self.cache.view(kvh, trace, n_reads=n)
                self._supply_unrotated_query(qi, heads)
                if self.softmax == "online":
                    y, st = self.attn.attend_online_batch(
                        qr[:, heads], kv, base=base, trace=trace)
                else:
                    y, st = self.attn.attend_causal_batch(
                        qr[:, heads], kv, base=base, tile=self.batch_tile, trace=trace)
                out[:, heads] = y
                report.attention.merge(st)
        else:
            for t in range(n):
                ctx = base + t + 1      # causal: this token sees itself and before
                for kvh in range(m.num_kv_heads):
                    heads = slice(kvh * m.kv_groups, (kvh + 1) * m.kv_groups)
                    self._supply_unrotated_query(qi[t], heads)
                    y, st = self._attend_one(qr[t, heads], kvh, ctx, trace)
                    out[t, heads] = y
                    report.attention.merge(st)
        report.context = base + n

        # 7. output projection, R^-1 already folded in -----------------------
        # The other side of the seam. This is what the FPGA hands back: Q16,
        # still in the rotated space, in `merge_heads` order. `to_float` and
        # `W_o'` on the next lines are the host's work again.
        merged = merge_heads(out)
        self._tap("out", merged)
        acc = self.qk_q.to_float(merged)
        y = project(self._output_proj(w), acc, unit="o_array", bias=w.o_bias,
                    chunk=self._chunk("o"), trace=trace)
        return y, report

    def _supply_unrotated_query(self, qi, heads) -> None:
        """Hand the attention the pre-rotation query. Architecture A only.

        A reconstructs cached keys into the original basis, so it has to score
        them against a query that was never rotated. B works entirely in the
        rotated domain and never reads this.
        """
        if self.compressed and self.attn.score_mode == "dense":
            self.attn._q_unrot = qi[..., heads, :]

    def _output_proj(self, w):
        """Which `W_o` the accumulator needs, which depends on its basis.

        B leaves the accumulator rotated and relies on `R^-1` having been
        folded into `W_o` offline. A has already rotated back, so it needs the
        original matrix. Getting this wrong does not raise -- it produces a
        model that runs and talks nonsense.
        """
        if self.compressed and self.attn.score_mode == "dense":
            if w.o_unrotated is None:
                raise ValueError(
                    "architecture A needs the unfolded W_o; these weights only "
                    "carry the folded one")
            return w.o_unrotated
        return w.o

    def _attend_one(self, q_rot, kv_head: int, ctx: int, trace: Trace | None):
        if self.compressed:
            kv = self.cache.view(kv_head, trace).select(slice(0, ctx))
            if self.softmax == "online":
                y, st = self.attn.attend_online(q_rot, kv, trace)
                return y.reshape(np.shape(q_rot)), st
            return self.attn.attend_two_pass(q_rot, kv, trace)
        return self._attend_dense(q_rot, kv_head, ctx, trace)

    def _attend_dense_batch(self, q_rot, kv_head: int, base: int,
                            trace: Trace | None, n_reads: int):
        """Every query token of a prefill against one KV head's fp16 cache.

        The float analogue of `attend_causal_batch`, and like it an evaluation
        strategy only: `n_reads` keeps the trace identical to the
        token-at-a-time path.

        Not bit-identical to that path, and it cannot be: BLAS blocks a batched
        GEMM differently from a sequence of GEMVs, so the float64 scores differ
        in their last bits and the fixed-point cast can land one count away.
        The compressed path is bit-identical, because its arithmetic is
        integer; this one is the fp32 baseline and is checked to a tolerance.
        """
        k, v = self.cache.view(kv_head, trace)
        for _ in range(n_reads - 1):
            record(trace, Op.MEM_READ, "memory",
                   n_bytes=self.cache.length * self.cache.bytes_per_token
                   // self.cache.n_kv_heads, cache="dense")
        k = k.astype(np.float64)
        v = v.astype(np.float64)
        n_tok = k.shape[0]
        qf = np.atleast_3d(self.qk_q.to_float(q_rot))
        n_q, heads, _ = qf.shape

        s = np.einsum("qhd,td->qht", qf, k)
        valid = (np.arange(n_tok)[None, :] <= (base + np.arange(n_q))[:, None])
        s = np.where(valid[:, None, :], s, -np.inf)
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        y = np.einsum("qht,td->qhd", p, v) / p.sum(axis=-1)[..., None]

        d = self.cfg.model.head_dim
        causal = heads * sum(base + i + 1 for i in range(n_q))
        record(trace, Op.SCORE, "attention", m=d, n=causal)
        record(trace, Op.ACCUMULATE, "attention", m=d, n=causal)
        return self.qk_q.from_float(y.reshape(np.shape(q_rot))), \
            AttentionStats(n_tokens=n_tok)

    def _attend_dense(self, q_rot, kv_head: int, ctx: int, trace: Trace | None):
        """The fp16 baseline, in the same rotated domain and the same Q format.

        Deliberately not a float softmax: if the baseline used different
        arithmetic, an accuracy gap would be attributable to the arithmetic
        rather than to the compression, which is the comparison being made.

        `q_rot` is (heads, d) -- the group sharing this KV head.
        """
        k, v = self.cache.view(kv_head, trace)
        k, v = k[:ctx].astype(np.float64), v[:ctx].astype(np.float64)
        qf = np.atleast_2d(self.qk_q.to_float(q_rot))
        s = qf @ k.T                                          # (heads, ctx)
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        y = (p @ v) / p.sum(axis=-1, keepdims=True)
        n_heads = qf.shape[0]
        record(trace, Op.SCORE, "attention", m=self.cfg.model.head_dim, n=ctx * n_heads)
        record(trace, Op.ACCUMULATE, "attention", m=self.cfg.model.head_dim, n=ctx * n_heads)
        return self.qk_q.from_float(y.reshape(np.shape(q_rot))), AttentionStats(n_tokens=ctx)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Start a new sequence. The weights and every table survive."""
        self.cache.reset()
