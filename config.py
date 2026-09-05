"""Configuration for the attention kernel.

One validated tree of frozen dataclasses. Nothing downstream may declare a
dimension, a bit-width or a latency of its own -- if a value is derivable it is
derived here exactly once, so a configuration cannot disagree with itself.

The split into four groups is deliberate and load-bearing:

    ModelConfig     what the network is.   Comes from the checkpoint.
    QuantConfig     how the KV cache is compressed.  An algorithm choice.
    FixedFormat     the fixed-point contract downstream of the arrays.
    HardwareConfig  what the accelerator looks like.  Affects CYCLES ONLY.

`HardwareConfig` is last because of a property the rest of the package
enforces: changing it must never change a numeric result. That is what lets the
same kernel run as a plain attention module inside a real model (timing off)
and as an accelerator model (timing on) without maintaining two code paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import ClassVar

from .ops.rope import RopeScaling


# --------------------------------------------------------------------------
# What the network is
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Shape of one attention block. Mirrors the fields an HF config exposes."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int | None = None      # None -> hidden_size // num_heads
    rope_theta: float = 10000.0
    max_position: int = 4096
    rope_scaling: "RopeScaling | None" = None   # None -> plain RoPE

    def __post_init__(self) -> None:
        d = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_heads
        object.__setattr__(self, "head_dim", d)

        if self.num_heads % self.num_kv_heads:
            raise ValueError(
                f"num_heads={self.num_heads} must be a multiple of "
                f"num_kv_heads={self.num_kv_heads} (grouped-query attention)"
            )
        if d & (d - 1):
            raise ValueError(
                f"head_dim={d} must be a power of two: the rotation is a "
                f"Walsh-Hadamard butterfly, which has no non-power-of-two form. "
                f"A dense d x d rotation would work numerically and has no "
                f"butterfly network, so it is out of scope for the architecture, "
                f"not merely unimplemented."
            )
        if d % 2:
            raise ValueError(f"head_dim={d} must be even (RoPE pairs channels)")

    @property
    def kv_groups(self) -> int:
        """Query heads sharing one KV head."""
        return self.num_heads // self.num_kv_heads

    @property
    def q_out_features(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_out_features(self) -> int:
        return self.num_kv_heads * self.head_dim

    @classmethod
    def from_hf(cls, hf_config) -> "ModelConfig":
        """Build from a transformers config object (or anything duck-typed).

        `rope_theta` moved: transformers <5 exposes it as a top-level attribute,
        >=5 nests it in a `rope_parameters` dict. Both are read, because reading
        neither and falling back to the 10000.0 default would silently serve a
        model whose positions are wrong -- fluent nonsense, not an error.
        """
        theta = getattr(hf_config, "rope_theta", None)
        if theta is None:
            params = getattr(hf_config, "rope_parameters", None) or {}
            theta = params.get("rope_theta")
        if theta is None:
            raise ValueError(
                "config exposes neither `rope_theta` nor "
                "`rope_parameters['rope_theta']`; pass ModelConfig directly "
                "rather than defaulting a value that decides token positions"
            )
        # "default" and "llama3" are modelled. Linear/NTK/YaRN are not, and are
        # refused rather than served on the default table: they change the angle
        # per position, so every token past the original context would be
        # misplaced -- fluent nonsense, not an error.
        params = getattr(hf_config, "rope_parameters", None) or {}
        if not params and isinstance(getattr(hf_config, "rope_scaling", None), dict):
            params = hf_config.rope_scaling
        rope_type = params.get("rope_type") or params.get("type") or "default"
        if rope_type == "llama3":
            scaling = RopeScaling.from_hf(params)
        elif rope_type == "default":
            scaling = None
        else:
            raise NotImplementedError(f"RoPE scaling {rope_type!r} is not modelled")

        return cls(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            head_dim=getattr(hf_config, "head_dim", None),
            rope_theta=float(theta),
            max_position=getattr(hf_config, "max_position_embeddings", 4096),
            rope_scaling=scaling,
        )

    # TinyLlama-1.1B, the shape this project has always been aimed at.
    @classmethod
    def tinyllama(cls) -> "ModelConfig":
        return cls(hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=64)


# --------------------------------------------------------------------------
# How the KV cache is compressed
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantConfig:
    """Rotate-then-quantize parameters for one KV head.

    `seed` selects the sign diagonal of the randomized Walsh-Hadamard
    transform. It is part of the format: a cache written under one seed is
    unreadable under another, so it travels with the weights, not with a run.
    """

    key_bits: int = 3
    value_bits: int = 2
    norm_bits: int = 16
    norm_frac: int = 9
    seed: int = 0
    symmetric: bool = True     # force codebook to c[i] == -c[n-1-i]

    # How many rounds of randomized Hadamard.
    #
    # ONE, on measured evidence. Two rounds was the default for a while on the
    # grounds that one overshoots into a platykurtic distribution -- which is
    # true, and turned out not to be what matters. Measured end to end on
    # WikiText-2 (12 windows, paired, TinyLlama at 1024 context), one round is
    # 1.3% BETTER in perplexity than two, and the difference is significant at
    # 2.8 standard errors. The likely mechanism is that each round ends in a
    # truncation, so a second round rounds the value off a second time for a
    # distributional gain the quantizer does not cash in.
    #
    # It is also half the hardware. See kernel/README.md, "Rounds".
    rot_rounds: int = 1

    # "gaussian" is the Lloyd-Max optimum for the unit normal, which is what
    # the rotation is supposed to produce and is why no calibration file is
    # needed. "uniform" is the MSE-optimal equally-spaced quantizer -- the
    # cheaper thing to build in hardware, and the control that says whether
    # the codebook shape is earning its place.
    codebook: str = "gaussian"

    def __post_init__(self) -> None:
        for name, b in (("key_bits", self.key_bits), ("value_bits", self.value_bits)):
            if not 1 <= b <= 8:
                raise ValueError(f"{name}={b} outside the supported range 1..8")
        if self.rot_rounds < 1:
            raise ValueError(f"rot_rounds={self.rot_rounds} must be at least 1")
        if self.codebook not in ("gaussian", "uniform"):
            raise ValueError(f"codebook={self.codebook!r}; expected gaussian or uniform")
        if self.norm_frac >= self.norm_bits:
            raise ValueError(
                f"norm_frac={self.norm_frac} leaves no integer bits in "
                f"norm_bits={self.norm_bits}"
            )

    def bytes_per_token(self, head_dim: int) -> int:
        """Wire size of one compressed KV token, for one KV head.

        Codes are bit-packed; the two norms are whole words. Raises rather than
        rounding if the codes do not land on a byte boundary -- a partially
        filled byte is a packing decision, and it belongs in the format, not in
        an implicit ceil.
        """
        code_bits = head_dim * (self.key_bits + self.value_bits)
        if code_bits % 8:
            raise ValueError(
                f"head_dim*(key_bits+value_bits) = {code_bits} is not a whole "
                f"number of bytes; choose widths summing to a multiple of "
                f"{8 // math.gcd(8, head_dim)}"
            )
        return code_bits // 8 + 2 * (self.norm_bits // 8)


# --------------------------------------------------------------------------
# The fixed-point contract
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedFormat:
    """Q-format widths for everything downstream of the systolic arrays.

    The arrays are floating point; the cache and the attention datapath are
    not. This is the contract at that boundary, and `numerics.fixed` is the
    only module allowed to act on it.
    """

    qk_frac: int = 16          # Q/K/V after the fp -> fixed conversion
    qk_width: int = 24
    centroid_frac: int = 15    # codebook entries, Q1.15
    acc_frac: int = 16
    acc_width: int = 32
    score_width: int = 48
    prob_frac: int = 15
    prob_width: int = 16
    recip_frac: int = 16
    exp_lut_bits: int = 8
    saturate: bool = True

    def score_shift_for(self, quant: QuantConfig) -> int:
        return self.qk_frac + self.centroid_frac + quant.norm_frac - self.acc_frac

    def acc_shift_for(self, quant: QuantConfig) -> int:
        """norm * prob * centroid  ->  Q(acc_frac)."""
        return self.centroid_frac + quant.norm_frac + self.prob_frac - self.acc_frac


# --------------------------------------------------------------------------
# What the accelerator looks like  (CYCLES ONLY -- never a numeric result)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ArrayConfig:
    """One systolic array.

    `rows` index the reduction dimension, `cols` the output dimension, which is
    the output-stationary assignment: each PE owns one partial sum and weights
    stream through. That choice is stated here rather than buried in the model
    because it is what decides utilisation on a single-token GEMV.
    """

    rows: int = 128
    cols: int = 128
    mul_latency: int = 1
    load_weight_cycles: int = 1

    @property
    def pe_count(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class RotateConfig:
    """`rot_fwht.sv`. One vector per cycle through six add/sub stages.

    `d = 64` makes `log2(d)` even, so the `1/sqrt(d)` scale is an exact right
    shift and the whole block is LUT -- no DSP, no reciprocal multiply. The
    throughput is therefore one vector per cycle regardless of `d`, and only
    the pipeline depth grows.
    """

    vectors_per_cycle: int = 1
    stage_latency: int = 1          # cycles per butterfly stage, log2(d) of them


@dataclass(frozen=True)
class EncoderConfig:
    """`rot_norm.sv` + `rot_encode.sv`. One rotated vector in, one cache row out.

    MEASURED, not estimated. Both blocks fold over the head dimension --
    `LANES = 8` of 64 channels per cycle -- because Phase A runs once per token
    against a Phase B that runs once per CACHED token. Sixty-four squarers and
    sixty-four comparator stacks would buy back cycles nothing is waiting on
    and take area from the score lanes, which are the thing that is actually
    bandwidth-limited.

    `cycles_per_vector` is `D/LANES + KEY_BITS + 1` = 13, the slower of the two
    blocks (the norm retires one vector every 9). `latency_cycles` is the norm's
    two accumulate stages plus its 24 square-root levels plus the encoder's
    KEY_BITS search levels and its write cycle.

    Both numbers come from RTL that closes at 250 MHz out of context
    (`plan.MD`, step 6): +0.931 ns for the norm, +2.298 ns for the encoder.
    The first attempt at each fused a stage too many and missed -- see the
    step 6 status block.
    """

    cycles_per_vector: int = 13
    latency_cycles: int = 31


@dataclass(frozen=True)
class AttentionConfig:
    """The score -> softmax -> accumulate lanes.

    `lanes` is the number of query heads scored against ONE cached row at a
    time. Under grouped-query attention the group sharing a KV head reads the
    same 52 B, so widening this costs adders and no bandwidth -- it is the one
    axis in the datapath that is free to parallelise, and the reason it is set
    to `kv_groups` rather than to 1.

    `rescale_cycles` is what a running-maximum update costs the scan: cycles in
    which a score is waiting and the datapath cannot take it. Charged per
    RESCALE record, which only the online softmax emits.

    MEASURED, not estimated. `tb_softmax_accum` counts the bubbles directly --
    a score sitting in the input FIFO with no pop -- over 39 rescale events
    across two stimulus sets, and gets **12 cycles each**, not the 1 an earlier
    draft assumed. A new maximum must not move while tokens computed against
    the old one are in flight, so the event is: drain the three-stage exp pipe,
    three more cycles to compute the factor from it, issue the SCALE, and wait
    while `accum` folds 64 channels through 8 multipliers.

    It was 11 until step 11 placed and routed `attn_top`. `accum`'s fold was
    one cycle -- select a group of 8 channels through a mux driven by all 64,
    then a 48x16 product -- and that path routed to -0.550 ns at 250 MHz with
    four lanes on the die, against +0.675 ns for the block alone. The fold is
    now two stages, so an event costs one cycle more. The arithmetic did not
    move: the goldens are unchanged and the bench still checks every one.

    It does not change any conclusion, and that is worth stating plainly. New
    maxima are logarithmically rare -- about ln T + 0.577 -- so at a context of
    32,768 this is 11 events and 132 cycles against a 32,786-cycle scan:
    **0.40%**, against the 0.27% the 8-cycle estimate gave. The rescale is
    still nearly free and the hazard still shrinks as context grows.

    `score_latency` covers S0..S10, score through softmax. `score_lane.sv`
    measures 8 cycles accept-to-score and `softmax_online.sv` adds 3 for the
    exp pipe, so 10 is now 11 -- it is drained once per scan and moves nothing.
    """

    lanes: int = 4                  # query heads in a group, in parallel
    score_latency: int = 11         # 8 in score_lane + 3 in the exp pipe
    rescale_cycles: int = 12        # measured; see the docstring


@dataclass(frozen=True)
class CacheConfig:
    """`kv_store_ddr.sv`: the DDR-resident cache, and the bandwidth it gets.

    THIS IS THE ONLY PLACE A MEASURED NUMBER ENTERS THE MODEL
    --------------------------------------------------------
    `ddr_gbps` is not an estimate. It is 14.708 GB/s, measured on the ZCU104 at
    4 AXI-HP ports, burst 64, outstanding 2, median of five repeats across two
    runs and a power cycle (`results/step1_ddr_sweep_run2_medians.log`). It is
    a read-only, four-stream, sequential figure -- the KV-scan pattern -- and
    must not be quoted as general memcpy bandwidth.

    The interface ceiling is derived, not stated: an AXI-HP port moves
    `port_bits/8` bytes per PL clock, so `hp_ports` of them cap the engine
    independently of what the DRAM can do. The model takes the smaller of the
    two and reports which one bound it, because the fix differs -- a short
    interface wants another port, a short DRAM wants a different access
    pattern.
    """

    lanes: int = 1                  # cache rows consumed per cycle
    parallel_kv: int = 1            # KV heads scanned concurrently
    hp_ports: int = 4
    port_bits: int = 128            # one AXI-HP master
    ddr_gbps: float = 14.708        # MEASURED. See the docstring.

    @property
    def interface_bytes_per_cycle(self) -> float:
        return self.hp_ports * self.port_bits / 8.0

    def ddr_bytes_per_cycle(self, clock_mhz: float) -> float:
        """The measured DRAM figure expressed in this clock's cycles."""
        return self.ddr_gbps * 1e9 / (clock_mhz * 1e6)

    def bytes_per_cycle(self, clock_mhz: float) -> float:
        return min(self.interface_bytes_per_cycle, self.ddr_bytes_per_cycle(clock_mhz))

    def bound_by(self, clock_mhz: float) -> str:
        return "interface" if self.interface_bytes_per_cycle <= \
            self.ddr_bytes_per_cycle(clock_mhz) else "dram"


@dataclass(frozen=True)
class MemoryConfig:
    """The off-die port, and the cost of using it."""

    port_bits: int = 256
    read_latency: int = 100     # cycles from request accepted to first beat
    write_latency: int = 8
    beat_gap: int = 0           # idle cycles BETWEEN beats once streaming
    weight_dtype_bytes: int = 2

    @property
    def bytes_per_beat(self) -> int:
        if self.port_bits % 8:
            raise ValueError(f"port_bits={self.port_bits} is not a whole number of bytes")
        return self.port_bits // 8

    def beats_for(self, n_bytes: int) -> int:
        return (n_bytes + self.bytes_per_beat - 1) // self.bytes_per_beat


@dataclass(frozen=True)
class HardwareConfig:
    q_array: ArrayConfig = field(default_factory=ArrayConfig)
    k_array: ArrayConfig = field(default_factory=ArrayConfig)
    v_array: ArrayConfig = field(default_factory=ArrayConfig)
    o_array: ArrayConfig = field(default_factory=ArrayConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    # The PL half of the seam. Named to match the `unit` strings the ops
    # already record, so `replay` reaches them by `getattr(hw, unit)` and a
    # trace naming a unit the hardware does not have is an error rather than a
    # silent zero -- which is what `ROTATE`, `QUANTIZE`, `SOFTMAX`,
    # `ACCUMULATE` and `RESCALE` used to be.
    rotate: RotateConfig = field(default_factory=RotateConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    clock_mhz: float = 100.0

    # Units the trace names that cost the PL nothing, and WHY -- an empty set
    # here would make every one of them a silent zero again.
    #
    #   rope   RoPE is applied on the host, before the wire. The seam is
    #          post-RoPE by construction (`plan.MD`, "Who owns what").
    #   codec  full dequantisation. The compressed path never calls it; only
    #          the dense baseline and the accuracy tests do.
    HOST_UNITS: ClassVar[frozenset[str]] = frozenset({"rope", "codec"})

    @classmethod
    def sized_for(cls, model: ModelConfig, side: int = 128, **kw) -> "HardwareConfig":
        """Arrays no wider than the projection they serve.

        K and V project to `num_kv_heads * head_dim`, which under grouped-query
        attention is `kv_groups` times narrower than Q. Giving all three the
        same array would leave the K and V arrays idle for most of their
        columns -- so the default sizes each one to its own output width, and
        the asymmetry shows up in the area report instead of hiding in the
        utilisation one.
        """
        q = ArrayConfig(rows=min(side, model.hidden_size), cols=min(side, model.q_out_features))
        kv = ArrayConfig(rows=min(side, model.hidden_size), cols=min(side, model.kv_out_features))
        o = ArrayConfig(rows=min(side, model.q_out_features), cols=min(side, model.hidden_size))
        # The score lanes default to the GQA group width: those query heads
        # share one cached row, so scoring them together costs adders and no
        # bandwidth. Sizing it from the model rather than defaulting to 4 keeps
        # a model without grouped-query attention honest -- there the group is
        # 1 and the lanes buy nothing.
        kw.setdefault("attention", AttentionConfig(lanes=model.kv_groups))
        return cls(q_array=q, k_array=kv, v_array=kv, o_array=o, **kw)

    @classmethod
    def zcu104(cls, model: "ModelConfig", **kw) -> "HardwareConfig":
        """The board this is being built for, with its MEASURED numbers.

        Two things are pinned here rather than left to defaults, because
        getting either wrong makes the report quietly optimistic:

        `clock_mhz = 250`. Implementation closed at 250 MHz on the step-1
        probe design (WNS +0.575 ns), and every bandwidth figure below is that
        clock's cycles.

        `memory.port_bits = 512`. The cache stream is four dedicated 128-bit
        AXI-HP masters, so the burst model and the bandwidth model must see the
        SAME interface -- 4 x 128 b = 64 B/cycle. Leaving `MemoryConfig` at its
        generic 256-bit default would have the burst model charging for a
        narrower port than the one the measurement was taken on, and it would
        show up as a burst bound where the truth is a DRAM bound.
        """
        kw.setdefault("clock_mhz", 250.0)
        kw.setdefault("cache", CacheConfig())
        cache = kw["cache"]
        kw.setdefault("memory", MemoryConfig(
            port_bits=cache.hp_ports * cache.port_bits, read_latency=100))
        return cls.sized_for(model, **kw)


# --------------------------------------------------------------------------
# The whole thing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelConfig:
    model: ModelConfig
    quant: QuantConfig = field(default_factory=QuantConfig)
    fmt: FixedFormat = field(default_factory=FixedFormat)
    hw: HardwareConfig | None = None       # None -> timing disabled entirely

    @property
    def timing_enabled(self) -> bool:
        return self.hw is not None

    @property
    def compressed_bytes_per_token(self) -> int:
        """One KV head's compressed token. Multiply by num_kv_heads for a layer."""
        return self.quant.bytes_per_token(self.model.head_dim)

    @property
    def dense_bytes_per_token(self) -> int:
        """The fp16 baseline this is measured against, one KV head."""
        return 2 * 2 * self.model.head_dim

    def with_hardware(self, hw: HardwareConfig | None) -> "KernelConfig":
        """Swap the hardware model. Guaranteed not to change any numeric result."""
        return replace(self, hw=hw)

    def describe(self) -> str:
        m, q = self.model, self.quant
        lines = [
            f"model      hidden={m.hidden_size} heads={m.num_heads} "
            f"kv_heads={m.num_kv_heads} d_head={m.head_dim} groups={m.kv_groups}",
            f"quant      key={q.key_bits}b value={q.value_bits}b "
            f"norm=Q{q.norm_bits - q.norm_frac}.{q.norm_frac} seed={q.seed}",
            f"per token  {self.compressed_bytes_per_token} B compressed vs "
            f"{self.dense_bytes_per_token} B dense, per KV head "
            f"({self.dense_bytes_per_token / self.compressed_bytes_per_token:.2f}x)",
        ]
        if self.hw is not None:
            lines.append(
                f"hardware   port={self.hw.memory.port_bits}b "
                f"lat={self.hw.memory.read_latency} clk={self.hw.clock_mhz:.0f}MHz"
            )
        else:
            lines.append("hardware   (timing disabled -- functional mode)")
        return "\n".join(lines)
