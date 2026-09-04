# A two-part attention block: host GPU + ZCU104 PL

## Context

`kernel/` is a numpy model of a compressed-KV attention block that is bit-exact
against `LlamaAttention` and instrumented to report what it would cost in
hardware. It has never run on hardware.

The analysis that led here:

- One Llama-3.2-1B attention block is **20 MiB of fp16 weights** against
  **4.75 MiB of on-chip memory** on the XCZU7EV. The block cannot be resident.
- At any realistic context, **weights dominate DDR traffic** — 96% at ctx 2048,
  crossing over to KV only near 50k tokens. A full-block FPGA implementation is
  a weight-streaming benchmark with an attention unit attached; the cache
  architecture, which is the actual contribution, would be 4% of the measured
  quantity.
- Splitting at RoPE puts **100% of FPGA DDR traffic in the KV cache**. Every byte
  and cycle measured on the board is then attributable to the thing being
  claimed.
- UG1267 p.35: the ZCU104's USB PHY is **host-only mode**, so the host link is
  Ethernet.

So: **two parts, one seam.** Part 1 is software on the host (projections, the
`1/√d` scale, the fp→fixed cast, RoPE, and `W_o'`). Part 2 is RTL on the PL:
rotate → quantize → cache → score → online softmax → accumulate.

**`R⁻¹` stays folded into `W_o` offline** (`ops/output.py:48`), as the package
already does by default. No inverse rotation exists anywhere in the system. This
deletes a hardware block, deletes a fixed-point truncation, and — the point that
matters most — **puts the seam on a value the unmodified package already
computes**: the FPGA returns exactly `merge_heads(out)` in Q16 from
`kernel.py:236`, just before `to_float` and the `W_o'` projection. Golden vectors
come from running `kernel/` as it stands, with no numeric changes that could
disturb the `cosine 1.000000` control in `single_layer.py`.

The forward rotation is still on the PL, which is what demonstrates the
randomized Hadamard in hardware. `W_o'` is 2048×2048 — identical in shape to
`W_o` — so the fold costs nothing at runtime.

**Target: long context, DDR-only.** The KV cache lives in the 2 GB PS DDR4 and
nowhere else. There is no URAM cache build. This is a deliberate narrowing: at
long context DRAM bandwidth is the wall, and a design that hides the cache in
URAM would be measuring the wrong thing.

**Deliverable: architecture spec + SystemVerilog RTL, in the `kernel/` repo.**

---

## The seam

Everything crossing is **Q8.16 fixed point packed into 24-bit lanes** — the
`FixedFormat` default, unchanged. The PL never sees a float, and the single
`to_fixed` cast stays in `ops/convert.py` on the host side.

| direction | contents | bytes/layer |
|---|---|---|
| down | q (32×64), k (8×64), v (8×64), post-RoPE | 9,216 |
| up | attention output (32×64), **rotated domain**, Q16 | 6,144 |

**245.8 kB per token** across 16 layers, in 16 sequential exchanges — layer *n+1*
cannot start until layer *n* returns.

24-bit lanes land exactly on a power-of-two bus because `d = 64`: 64 values ×
24 bits = 1,536 bits = **3 beats of 512**. No padding, no partial beats, no
`TKEEP` decoding, no DRE. (Same power-of-two property that makes the FWHT's
`1/√d` an exact shift by 3.)

The bus is **512-bit AXI4-Stream**. Widths are enumerated by the DMA IP —
`axi_dma_v7_1/component.xml:6047` allows 8/16/32/64/128/256/512/1024 and nothing
else, so a literal 24-bit bus is impossible at the memory boundary even though it
is legal AXI4-Stream.

**Open decision: fp16 on the wire.** Halves the bytes (245.8 → 163.8 kB/token) at
the cost of moving `to_fixed` onto the PL, so `ops/convert.py` becomes a *model*
of the hardware cast rather than the cast itself. At the long contexts this plan
targets the wire is only ~12% of step time, so this is **low priority** — but
settle it before the format is frozen, and never stack it with 24-bit packing.

### Why Q8.16, with the headroom stated

What crosses the wire is **pre-rotation**, so it is not yet Gaussianised — the
rotation is the PL's first stage. Measured on the real checkpoint, 400 tokens,
all 16 layers:

| | max abs |
|---|---|
| q (already ×1/√d) | 2.85 |
| **k** | **21.06** |
| v | 4.62 |

The k outliers are exactly the pathology the randomized Hadamard exists to fix,
so the wire must carry them at full range. That headroom is deliberate.
Narrower lanes (22 bits, Q6.16) would save 8% of the wire while losing 512-bit
alignment and cutting headroom on the strength of one 400-token passage;
re-splitting to Q7.17 is a real numerics change (`qk_frac` feeds `score_shift`
and `acc_shift`, `config.py:225`) for a gain that is near zero next to 4-bit key
quantisation.

**Step 4 measured this on 20× the text, and it moved.** 8,188 tokens of
WikiText-2, all 16 layers, 402,653,184 elements through the cast: **max abs
24.957, saturation rate 0**. The peak is 19% above the 400-token figure — which
is what more text should do, and is exactly why the gate runs before the format
is frozen rather than after. So the true headroom against ±128 is **5.13×**,
not the 6.1× an earlier draft computed from 21.06, and the 22-bit lane would
have been left at ~1.3× rather than 1.5×.

**Q8.16 stays, now on measured evidence** — it is the `FixedFormat` default, so
RTL and golden model share one number.

### Who owns what

Host keeps: embeddings, RMSNorm, the four projections, the `1/√d` scale,
`to_fixed`, RoPE, **`W_o'` with `R⁻¹` folded in**, residual, MLP.
PL owns: rotate, quantize, cache, score, online softmax, accumulate.

**No QK-norm.** The kernel supports it (`kernel.py:167`, fires only when
`w.q_norm is not None`) because it is model-agnostic, but Llama 3.2 1B does not
have it — the checkpoint's layer 0 carries only `input_layernorm`,
`post_attention_layernorm` and the four projections. Do not build for it.

**The `1/√d` scale stays on q, on the host.** `(q·k)/√d` and `(q/√d)·k` are the
same number, but scaling q is 64 multiplies per head per token while scaling the
score is one per cached token. More importantly, **the PL score path is
deliberately multiplier-free**; a multiply inside that loop would destroy the
property the design is built around. `1/√64 = 1/8` is an exact shift by 3. It is
applied *after* the norm (`kernel.py:172`) because RMSNorm erases input scale.

`W_o'` is **seed-specific** — it bakes in the `R` chosen by `QuantConfig.seed`,
so changing the seed means re-folding. Already true of the cache format
(`config.py:135`). For debugging, `Rotation.matrix()` un-rotates the FPGA's raw
output on the host in one line.

---

## PL microarchitecture

Parameters: `d=64`, `KV_HEADS=8`, `Q_HEADS=32`, `GROUPS=4`, `KEY_BITS=4`,
`VAL_BITS=2`, **`PARALLEL_KV=1`**. Q-formats from `config.py:213`: q/k/v Q8.16 in
24 bits, centroids Q1.15, norms Q7.9 in 16 bits, probabilities Q15 in 16 bits,
accumulator Q16 in 32 bits, scores Q16 saturating in 48 bits.

### The sizing result: `PARALLEL_KV = 1`, and why nothing larger

One lane group consumes **52 B/cycle = 13.0 GB/s** at 250 MHz. PS DDR4-2400 gives
19.2 GB/s peak; **measured at 14.7 GB/s** (step 1), so the engine runs with 13%
headroom.

| `PARALLEL_KV` | bandwidth needed | result |
|---|---|---|
| **1** | **13.0 GB/s** | **matched — no waiting** |
| 2 | 26.0 GB/s | starves, engine idle ~50% |
| 4 | 52.0 GB/s | starves badly |

**Anything above 1 is dead silicon on a DDR-only build.** The balance point is
1.03 lanes at realistic DDR throughput, 1.48 at optimistic peak.

**Four AXI-HP ports are required.** A 128-bit HP port moves 16 B per PL clock,
so at a single 250 MHz domain it is **4.0 GB/s**, not the 5.33 GB/s an earlier
draft assumed (that figure was 333 MHz, i.e. a second clock domain and a CDC):

| ports @ 250 MHz | aggregate | vs 13.0 GB/s needed |
|---|---|---|
| 3 | 12.0 GB/s | **short** |
| **4** | **16.0 GB/s** | 23% headroom |

Four ports, one clock domain, no CDC. The headroom is the point — 16.0 GB/s of
interface against 13.0 GB/s of appetite leaves room for the DRAM controller to
fall short of its own peak, which it certainly will. A single port caps the
engine at 31%.

### Ingress: 8 groups of 6 vectors, and it is O(1) in context

The host sends **grouped by KV head**, so the block starts on group 0 after
1,152 bytes instead of 9,216:

```
group 0: [ k₀ ][ v₀ ][ q₀ ][ q₁ ][ q₂ ][ q₃ ]   1,152 B
group 1: [ k₁ ][ v₁ ][ q₄ ][ q₅ ][ q₆ ][ q₇ ]   1,152 B
   ...                                          ───────
group 7: [ k₇ ][ v₇ ][ q₂₈…q₃₁ ]         total  9,216 B
```

**k and v must precede the q's inside a group — causality, not preference.** The
current token attends to itself, so its own key and value must be in the cache
before any score is computed. `cache.append_rotated` at `kernel.py:187` runs
*before* attend, which then selects `slice(0, ctx)` including the new token.
Getting this order wrong produces an off-by-one-token attention that still runs.

Each arriving 1,536-bit vector is consumed immediately and never held raw at full
width:

| arriving | becomes | change |
|---|---|---|
| **k** | 256 b codes + 16 b norm = **34 B** | 5.6× shrink |
| **v** | 128 b codes + 16 b norm = **18 B** | 10.7× shrink |
| **q** | rotated, held raw at 24 b | unchanged until needed |

**Accept all 8 groups immediately; do not backpressure.** k and v go straight to
their own DDR write buffers — group *h+1* writes head *h+1*'s region while group
*h* reads head *h*'s, so they never collide. The 32 rotated q vectors are held
**raw at 6 KiB**, and each group's product table is built **just-in-time**
(64 cycles against a `T`-cycle scan, ping-ponged so it never enters the critical
path). Expanding all eight up front would cost 160 KiB — 27× more — and is what
would have forced backpressure.

**Ingress cost is constant in context.** At ctx 512 or ctx 131,072 the block
receives the same 8 groups of 6 vectors per token. Only the scan lengthens.

### The result that shapes the datapath

**The inner loop has no multipliers in it.** Both loops reduce to table lookups
and adds, and this falls directly out of attending on codes:

- **Score.** `<q_rot, centroid[code]>` where `centroid` has only `2^4 = 16`
  possible values. Precompute, once per query vector, the 64×16 table of
  `q_rot[ch] × c_i`. Every cached token afterwards is 64 gathers and a 64-input
  adder tree, amortised over the whole context.
- **Accumulate.** `v_norm · p · centroid_v[code]` where `centroid_v` has only
  `2^2 = 4` values. Compute `w = v_norm × p` (one DSP), then `w × c_v` for the
  four constants (four DSPs); the 64 channels are a 4:1 mux and an add.

Three DSPs in the inner loop where a conventional attention unit needs 64. Four
lanes cost ~12 DSP and ~12k LUT against 1,728 DSP and 230k LUT — which is why
**DRAM, never arithmetic, is the constraint.**

### Dataflow: three phases per token

```
 PHASE A — once per token                      ~150 cycles
 ┌────────────────────────────────────────────────────────┐
 │ 48 vectors in (32 q, 8 k, 8 v), 64 values each         │
 │   rotate  : FWHT-64  ─ 1 vector/cycle ─ 48 cyc         │
 │   quantize: k,v only ─ norm + codes   ─ 16 cyc         │
 │   write   : 8 cache rows × 52 B → DDR                  │
 │   hold    : 32 rotated q raw, 6 KiB                    │
 └────────────────────────────────────────────────────────┘
                            ↓
 PHASE B — once per CACHED TOKEN               8 × T cycles   (>99%)
 ┌────────────────────────────────────────────────────────┐
 │ for kv_head in 0..7:              ← 8, SEQUENTIAL      │
 │   build this group's product tables (64 cyc, hidden)   │
 │   for t in 0..T-1:                ← T, SEQUENTIAL      │
 │     stream one 52 B row from DDR                       │
 │     for j in the 4 query heads:   ← 4, PARALLEL        │
 │       score → softmax → accumulate                     │
 └────────────────────────────────────────────────────────┘
                            ↓
 PHASE C — once per token                      ~64 cycles
 ┌────────────────────────────────────────────────────────┐
 │ 32 reciprocal divides, acc × recip >> 16, egress       │
 └────────────────────────────────────────────────────────┘
```

At ctx 32,768 Phase B is 262,144 cycles against ~200 for A and C. **Optimise
nothing but Phase B.**

### The four axes

| axis | extent | parallel? | why |
|---|---|---|---|
| **channels** `d` | 64 | **fully parallel** | independent; adder tree reduces in 6 levels |
| **query heads in a group** | 4 | **parallel, 4 lanes** | they share the *same* cached row — the GQA win |
| **KV heads** | 8 | **sequential** | `PARALLEL_KV=1`; DRAM cannot feed more |
| **cached tokens** `T` | T | **sequential** | the online softmax carries state |

### One score lane, stage by stage

```
 S0   stream 52 B row for token t from DDR
 S1   split → 64 k-codes (4b) | 64 v-codes (2b) | k_norm | v_norm
 S2   64 PARALLEL table reads:  prod[ch] = qtab[ch][ kcode[ch] ]      ← no multiplier
 S3 ┐
 ..  │ adder tree, 64 → 1, six levels                                  ← no multiplier
 S8 ┘
 S9   × k_norm                                        2 DSP
 S10  >> 24, clamp                    → score s_t  in Q16
 S11  compare vs m ; exp LUT          → p          in Q15   ─┐
 S12  l += p ;  w = v_norm × p              1 DSP            │ loop-carried,
 S13  64 PARALLEL: acc[ch] += (w × c_v[vcode[ch]]) >> 23     │ 3 deep
                            └─ only 4 possible c_v → a mux, ─┘
                               not 64 multipliers
```

Throughput 1 result/cycle, latency ~13. Four lanes share one row.

### The softmax recurrence, and why it is nearly free

`m`, `l` and `acc` are loop-carried. But **`m` only changes when a new maximum
appears, and new maxima are logarithmically rare** — expected count in T samples
is the harmonic number, ≈ ln T + 0.577:

| ctx | expected rescales | cost (8 DSP × 8 cyc) | % of scan |
|---|---|---|---|
| 8,192 | ~9.6 | 77 cyc | 0.94% |
| 32,768 | ~11.0 | 88 cyc | **0.27%** |
| 131,072 | ~12.4 | 99 cyc | 0.08% |

**Design: speculate "no new max."** Between rescales `m` is constant, so `exp`,
`l +=` and `acc +=` are pure feed-forward and pipeline at II=1. Compute
`p = exp(m_{t−1} − s_t)` speculatively alongside the comparison; commit on the
common path, flush and rescale on the rare one. **The hazard shrinks as context
grows** — which is the regime this plan targets.

Each of the 4 lanes has its own `m`, `l`, `acc`, so the recurrences are
independent. Size the rescale DSPs from `AttentionStats.rescale_events` measured
on real text, not from the harmonic bound.

**Considered: two-pass softmax.** It removes the recurrence entirely and would be
bit-exact against `forward` (which uses two-pass) rather than `attend_online`. It
costs 2× DDR reads — and DDR is the wall. Rejected on that basis, not because
online is obviously better.

### Contention and buffering

Phase A of group *h+1* overlaps Phase B of group *h*, on nearly disjoint hardware:

| hardware | arriving (A) | computing (B) |
|---|---|---|
| ingress unpack, `rot_fwht`, `rot_norm`, `rot_encode` | ✔ | |
| DDR **write** path, table build | ✔ | |
| DDR **read** stream, score lane | | ✔ |
| **exp LUT, softmax, accumulator** | | ✔ |

**The softmax is never touched by an arriving group**, and only one group is ever
in Phase B. No cross-group contention.

Two real contention points:

1. **exp LUT.** Four lanes each need a lookup per cycle. **Replicate 4×**:
   68 Kb × 4 = 272 Kb ≈ 8 BRAM36. Do not share one.
2. **Rescale bubbles.** A **16 × 48 b elastic FIFO** between S10 and S11 keeps
   the 10-stage score pipe running at 1/cycle through a rescale stall.

### On-chip memory budget

URAM is **not** used for the cache. It buffers the DDR stream.

| item | size |
|---|---|
| DDR prefetch / tile buffer (double-buffered, ~64 KB) | 512 Kb — 2 URAM |
| exp LUT, 4353 × 16 b (`ops/attention.py:110`), ×4 lanes | 272 Kb — 8 BRAM36 |
| q×centroid product tables, 4 heads × 64 × 16 × 40 b, ×2 ping-pong | 320 Kb — 1.1 URAM |
| raw rotated q, 32 × 64 × 24 b | 6 KiB — 2 BRAM36 |
| accumulators, 4 heads × 64 × 32 b | 1 KB — registers |
| score → softmax elastic FIFO | < 1 Kb |
| codebooks, boundaries, sign ROMs | < 4 Kb |

Under 1 Mb of 38 Mb available. The accumulator is only 1 KB because groups run
one at a time, so four heads' state is live rather than thirty-two.

### `rot_fwht.sv` — where the geometry pays off

`d = 64`, so `log2(d) = 6` is even and `sqrt_shift(64) = 3`
(`numerics/fixed.py:204`). The `1/√d` scale is an **exact right shift by 3** — no
multiplier, no `inv_sqrt_q15` path. Six stages of 32 add/sub butterflies plus a
sign flip and a shift: pure LUT, zero DSP.

**Forward only** — `R⁻¹` is folded into `W_o'` on the host. No inverse path, no
mode bit, no extra truncation. One instance shared across q, k and v.

Mirror `ops/rotate.py:127` exactly, including the `sqrt_shift` /
`inv_sqrt_q15` split — `d=64` takes the exact-shift branch, but the parameter
should make the other branch a compile error rather than silently wrong.

### `kv_store_ddr.sv` — the only cache

Row format must match `ops/quantize.py:407` byte for byte: per (token, kv head),
LSB-first `[k codes | v codes | k_norm LE16 | v_norm LE16]` = 32 + 16 + 2 + 2 =
**52 B** at 4b/2b. Packed tight in DDR; the address of head *h*, token *t* is a
flat stride so the prefetcher issues long sequential bursts.

Requirements:

- **4 AXI-HP master ports**, one per port dedicated (no shared interconnect --
  funnelling them through one smartconnect serialises them at the crossbar and
  measures the interconnect, not the DRAM), sustaining 13.0 GB/s aggregate.
- **Deep prefetch.** Every layer streams its whole cache; the access pattern is
  perfectly sequential and known one full scan ahead, so latency is entirely
  hideable — but only with enough outstanding transactions. Size the queue from
  the measured DDR latency, not from a guess.
- **Separate write path** for the new token's 8 rows, 416 B/token/layer.
  Negligible traffic, but it must not stall the read stream.

### Blocks

| module | function |
|---|---|
| `attn_ingress.sv` | 512-bit AXI4-Stream unpack; 24-bit lanes straddle beats, so a shift register and counter |
| `rot_fwht.sv` | randomized Hadamard, forward only |
| `rot_norm.sv` | RMS norm → Q7.9 wire norm; integer sqrt, `ops/quantize.py:215` |
| `rot_encode.sv` | threshold compare → codes; binary search, `KEY_BITS` stages |
| `kv_store_ddr.sv` | 4× AXI-HP master, prefetcher, write path |
| `qtab_build.sv` | just-in-time product tables, ping-pong |
| `score_lane.sv` | product-LUT gather + adder tree + norm multiply, ×4 |
| `softmax_online.sv` | running max/sum, exp LUT, speculative commit, rescale |
| `accum.sv` | 4-constant mux accumulate, 64 channels |
| `finalize.sv` | reciprocal divide, `acc × recip >> 16` |
| `attn_top.sv` | control FSM, AXI4-Stream wrapper, `sa_top.sv` pattern |

**`attn_top.sv` must stream through, not store and forward.** Begin on group 0 as
soon as it lands; return each group's 4 output heads as they finalize.

### Softmax must be the online form

`ops/attention.py:283` is the hardware order; `forward` uses two-pass. The README
is explicit that the two are **not** bit-identical and must not be asserted to be
— the online form truncates the accumulator once per rescale. **The testbench
golden model is `attend_online`, not `attend_two_pass`.** Getting this backwards
produces a testbench that can never pass.

---

## Implementation plan

Fifteen steps. **Each step ends with a command that passes.** Do not start step
*n+1* until step *n*'s exit criterion is green — that is the loop.

Two rules that hold throughout:

- **No numeric path changes.** `python -m kernel.tests.run` and the
  `cosine 1.000000` control in `single_layer.py` must pass, unchanged, at every
  step. If they move, cost modelling has leaked into the datapath.
- **Every RTL step ends with out-of-context synthesis.** Resource and timing
  numbers in this plan are estimates until `run_synth.sh` says otherwise.

### Phase 0 — kill the riskiest assumption first

Everything in this design was sized from an assumed "DDR sustains ~13.4 GB/s."
Measure it **before** writing a line of attention RTL. (Done — it delivers 14.7,
and the prefetch depth the sweep found is the opposite of the obvious one.)

**Step 1 — DDR bandwidth probe.**
A minimal PL design: 4 AXI-HP masters issuing long sequential reads over a
multi-MB region, with a cycle counter and an AXI-Lite status block. No attention
logic. Reuse `systolic_zcu104`'s block-design and `xsdb` bring-up flow verbatim.
Sweep at runtime from one bitstream: 1/2/3/4 ports x burst 16/64/128/256 x
outstanding 1..32.
*Files:* `rtl/axi_rd_engine.sv`, `rtl/ddr_probe.sv`, `tb/tb_ddr_probe.sv`,
`scripts/regress.sh`, `scripts/create_bd_probe.tcl`, `sw/probe_main.c`
*Exit:* a table of achieved GB/s vs configuration, printed over UART.
**Gate:** if sustained < 13.0 GB/s, revise `PARALLEL_KV`, the clock, or the
KV format **before proceeding**. Record the measured number in this document.

**Status: MEASURED ON HARDWARE. GATE PASSES.**

Implementation closed at 250 MHz (WNS +0.575 ns, WHS +0.017 ns, 0 errors).
Two independent runs across a power cycle; run 2 repeats every configuration
five times and reports the median. Logs in `results/`.

**Measured: 14,708 MB/s at 4 ports, burst 64, outstanding 2.**
Spread over 5 repeats: 14,697–14,734 (**0.25%**). Run 1 independently saw
14,678. That is **77% of the 19.2 GB/s DDR4-2400 peak** and 92% of the 4-port
interface ceiling.

| | |
|---|---|
| engine appetite, `PARALLEL_KV=1`, 4b/2b @ 250 MHz | 52 B/cyc = **13.0 GB/s** |
| measured DDR | **14.7 GB/s** |
| headroom | **13%** |

`PARALLEL_KV=1` at 250 MHz with 4b/2b is fed. No clock reduction and no format
change are needed.

### The prefetcher must be SHALLOW: burst 64, outstanding 2

The most useful result in the sweep, and a counterintuitive one. At 4 ports,
burst 64:

| outstanding | MB/s | spread over 5 repeats |
|---|---|---|
| 1 | 8,458 | 8,455–8,461 |
| **2** | **14,708** | **14,697–14,734** |
| 4 | 11,987 | 11,752–12,130 |
| 8 | 11,870 | 11,845–12,501 |
| 16 | 11,812 | 11,487–12,381 |
| 32 | 11,947 | 11,389–12,044 |

**A deeper queue costs 19% of bandwidth.** Depth 1 cannot hide latency; depth 2
hides it; depth 4 and beyond make things worse and *noisier* (0.25% spread at
depth 2 against 3–6% above it). The mechanism is row-buffer locality: the four
ports read regions 64 MB apart, so a deep queue interleaves requests across four
different DRAM rows and thrashes activate/precharge, while a shallow queue lets
each stream keep its row open. The rising variance is the fingerprint — thrashing
is timing-dependent in a way that clean sequential access is not.

So size `kv_store_ddr` at **burst 64, outstanding 2 per port, and no deeper**.
The instinct to build a deep prefetch queue is actively wrong here, and only a
measurement could have said so.

**A correction worth recording.** After run 1 I called the 14,678 point an
artifact, on the grounds that a deeper queue cannot reduce throughput. The
reasoning was wrong — it can, for the row-locality reason above — and the
repeats showed the opposite of what I expected. The repeat-and-take-the-median
change was still the right move: it is what turned a number I did not trust into
one I do.

**Ports: 4 is correct, and for a reason that does not depend on the clock.**
One 128-bit port carries 16 B/cycle, so the count is `ceil(row_bytes / 16)`:
4b/2b's 52 B needs 4. (3 ports peaked at 11,797 MB/s against their 12,000
ceiling — interface-limited, and short of the 13.0 needed.)

### Phase 1 — golden model and tooling (Python only)

**Step 2 — cost model.**
`RotateConfig`, `EncoderConfig`, `AttentionConfig`,
`CacheConfig(lanes, parallel_kv, hp_ports, ddr_gbps)` next to `ArrayConfig`
(`config.py:238`); fields added to `HardwareConfig` (`config.py:278`). New
`hw/attn_block.py`. Wire the units through `getattr(hw, unit)` in `replay.py:262`
the way `MATVEC` already is — today `ROTATE`, `QUANTIZE`, `SOFTMAX`,
`ACCUMULATE`, `RESCALE` cost **zero cycles** and `SCORE` is a flat `sum(w.n)`
(`replay.py:272`). Add the DDR bandwidth bound, seeded with step 1's measurement.
Update `serial_cycles` (`:209`), `critical_path_cycles` (`:214`), and
`test_hardware.py::check_replay_produces_both_bounds`, which asserts the exact
unit set (`:204`).
*Exit:* `python -m kernel.tests.run` green, including a new
`check_ddr_bandwidth_is_the_reported_bound_at_long_context`.

**Status: DONE.** `hw/attn_block.py` holds `RotateUnit`, `EncoderUnit`,
`AttentionUnit` and `CacheBandwidth`; `config.py` holds the four configs plus
`HardwareConfig.zcu104()`, which pins 250 MHz and sets `memory.port_bits` to
`4 × 128` so the burst model and the bandwidth model see the same interface.
`replay` now resolves every unit by `getattr(hw, unit)`, and a unit that costs
the PL nothing must be listed in `HardwareConfig.HOST_UNITS` with its reason —
`rope` (host-side by the split) and `codec` (the compressed path never
dequantises). Anything else raises.

Two modelling decisions worth stating, because both change the headline:

- **SCORE and ACCUMULATE are the max, not the sum.** They are the same
  (query, cached token) pairs one pipeline stage apart. Summing them would
  double the dominant term of a long-context step, for the same reason weight
  bytes are not charged to both the arrays and the memory model.
- **The cache has two bounds and both are reported.** `MemoryModel` charges
  bursts and latency; `CacheBandwidth` charges the measured 14.708 GB/s.
  `memory_cycles` is the max and `memory_bound_by` says which. They cross over:
  at ctx 8 it is bursts, at ctx 4096 it is bandwidth.

At ctx 4096 the model reproduces the microarchitecture's own arithmetic without
being told it: attention retires `8 × 4097` cycles — 8 KV heads sequential, 4
query heads parallel — against 78 cycles for Phase A. 124 checks green.

**Step 3 — vector generator.**
`hw/vectors.py`, in the style of
`fpga_work/systolic_zcu104/scripts/gen_vectors.py`. Emits the 24-bit packed beat
stream and the expected `merge_heads(out)` in Q16 as hex, tapped from
`kernel.py:236` via a debug callback — not a datapath change. Also emits
per-stage intermediates (post-rotate, post-quantize, per-token scores) so each
RTL block below has its own vectors.
*Exit:* `check_24bit_packing_round_trips` green; vectors regenerate
deterministically from a seed.

**Status: DONE.** `hw/vectors.py` plus `tests/test_vectors.py` (11 checks).
`AttentionKernel.on_stage` is the debug callback — an observer with the same
`is not None` guard the trace uses, taking `wire.q/k/v` (the seam), `rot.q/k/v`
and `out` (`merge_heads` in Q16, before `to_float` and `W_o'`).
`check_the_tap_is_an_observer_and_leaves_no_trace` asserts a step run with the
observer attached is bit-identical to one without.

Emitted at ctx 64: **144 beats** of ingress (48 vectors × 3 beats, 9,216 B, 8
groups of 1,152 B) and **52 B × 8** cache rows — the plan's numbers, produced
rather than restated. `out.hex` (two-pass, what `forward` returned) and
`out.online.hex` (the hardware order) are both written, and
`manifest.json:softmax_golden` names the online one, because a testbench
pointed at the other can never pass. A check asserts the two differ — if they
ever agree, the online path has stopped rescaling.

The packing round trip is tested at every start alignment and at 2/3/4/9/16/17/24
bits, not just at 24: a lane straddles a 512-bit beat boundary, and an
off-by-one shift produces a stream of exactly the right length full of
plausible values.

**Step 4 — saturation gate.**
`python -m kernel.experiments.wikitext_ppl`, read
`report.conversion.saturation_rate` and `max_abs_float`.
*Exit:* **`saturation_rate == 0`** on a sample wider than the 400-token passage.
`saturate=True` clamps rather than wraps, so a non-zero rate is silent quality
loss. If it fires, the wire format changes and steps 5–11 shift with it — which
is why this comes before any RTL.

**Status: MEASURED. GATE PASSES.**

`wikitext_ppl.py` did not report these figures; it now aggregates
`StepReport.conversion` over every grafted layer and carries `sat` and `max|x|`
in the table, accumulated across windows rather than per window — a rate that
is zero on window 1 and non-zero on window 40 is exactly what this is for. The
stats were already computed inside every `forward` and were being discarded at
the adapter boundary, so this costs nothing.

Llama 3.2 1B (`models/Llama-3.2-1B`), WikiText-2 test, 4 non-overlapping
windows of 2,048 = 8,188 tokens, k4v2, all 16 layers:

| | |
|---|---|
| elements through the fp → fixed cast | **402,653,184** |
| **saturated** | **0** |
| max abs float | **24.957** (Q8.16 headroom **5.13×**) |
| ppl | 12.71 |
| cache | **416 B/token/layer** — the plan's figure, measured |

**24-bit Q8.16 lanes are locked.** The wire format is settled and steps 5–11 can
proceed against it. Note the peak is 19% above the 400-token figure the width
was originally argued from, so the real headroom is 5.1×, not 6.1× — still
ample, and the reason the narrower 22-bit lane was right to reject.

### Phase 2 — datapath blocks, bottom-up

Each has its own self-checking testbench against step 3's vectors.

**Step 5 — `rot_fwht.sv`.** Forward only. Six stages of 32 add/sub butterflies,
sign flip, `>>3`. Pure LUT, zero DSP. Mirror `ops/rotate.py:127` including the
`sqrt_shift` / `inv_sqrt_q15` split — make the non-exact branch a compile error.
*Exit:* `tb_rot` bit-exact on ≥1000 random vectors; synth clean at 250 MHz.

**Status: DONE.** `rtl/rot_fwht.sv` (forward only, six levels of 32 add/sub
butterflies, the seed-0 sign diagonal as a `SIGN_NEG` parameter, exact `>>> 3`),
`tb/tb_rot.sv`, `scripts/synth_rot_fwht.tcl` + `run_synth.sh`.

The non-exact branch is a compile-time absence rather than a runtime check: the
module hard-codes `SHIFT = 3` for d = 64 and documents that any other dimension
belongs on `ops/rotate.py`'s reciprocal-multiply path. Sign extension happens
*before* the diagonal negation, because `-24'sh800000` is +2^23 and does not fit
the input width; the internal width is 31 b (24 + 6 butterfly levels + the
negation endpoint). A `range_error` output reports a normalized result that
would not fit 24 bits — it never silently wraps or saturates behind the numeric
model.

Verified at **1,072 vectors**: the 48 real q/k/v tensors tapped from the
unmodified kernel plus 1,024 bounded random vectors, all golden from
`Rotation.apply`. The bench also asserts the DUT's `SIGN_NEG` equals the Python
artifact's sign word (a lane-order or PCG64 mismatch fails there, not 64
butterflies later), streams at full rate to confirm one vector retires per cycle
after the six-stage fill, and holds `out_ready` low to confirm a stalled
consumer freezes the whole pipe with the output stable.

Out-of-context synthesis, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| | |
|---|---|
| WNS | **+3.309 ns** (4.0 ns period) |
| LUT | 12,385 |
| FF | 11,361 |
| **DSP** | **0** |
| BRAM/URAM | 0 |

Zero DSP, as the geometry promised — the whole rotation is add/subtract and a
shift. 3.3 ns of slack at 250 MHz says the six levels are registered finely
enough that this block will not be the critical path in `attn_top`.
`./scripts/regress.sh` green on both benches; `python -m kernel.tests.run` 124
passed, unchanged.

**Step 6 — `rot_norm.sv` + `rot_encode.sv`.** Integer sqrt
(`ops/quantize.py:215`) and the binary-search threshold compare, `KEY_BITS`
stages.
*Exit:* codes and norms **byte-exact** against `quantize.encode`, including the
`>` tie-to-lower-bin rule; row layout matches `ops/quantize.py:407` byte for byte.

**Step 7 — `qtab_build.sv` + `score_lane.sv`.** The multiplier-free core: 64
parallel table gathers, six-level adder tree, one `k_norm` multiply, shift and
clamp. Ping-pong tables.
*Exit:* scores bit-exact against `CompressedAttention.scores` for a fixed cached
row; synth confirms ~3 DSP and ~3k LUT per lane.

**Step 8 — `softmax_online.sv` + `accum.sv`.** Speculative "no new max" commit,
flush-and-rescale on the rare path, 16×48 b elastic FIFO, 4-constant mux
accumulate. **exp LUT replicated ×4.**
*Exit:* bit-exact against `attend_online` — **not** `attend_two_pass` — at short
context, where rescales are frequent enough to exercise the flush path.

**Step 9 — `finalize.sv`.** One reciprocal divide per query head, `acc × recip
>> 16`.
*Exit:* end-of-scan output bit-exact against `_finalize`
(`ops/attention.py:553`).

### Phase 3 — memory and integration

**Step 10 — `kv_store_ddr.sv`.** 4× AXI-HP master, one per port; deep prefetch sized
from step 1's measured latency; separate non-blocking write path for the new
token's 8 rows. **The highest-risk block** — the only one whose correctness and
performance depend on a memory system rather than on vectors.
*Exit:* against a DDR behavioural model, sustains the step-1 bandwidth with the
score pipe never starved; row round-trip byte-exact against `quantize.pack`.

**Step 11 — `attn_top.sv`.** Control FSM, 512-bit AXI4-Stream wrapper, 8 groups
of 6 vectors, k/v before q, raw-q hold (6 KiB) with just-in-time table build,
stream-through egress.
*Exit:* `tb_attn` bit-exact for multi-token, multi-context runs with
back-pressure; full `scripts/regress.sh` green; place-and-route meets 250 MHz
with positive WNS.

### Phase 4 — hardware bring-up

Mirror the order that already worked in `systolic_zcu104`.

**Step 12 — JTAG.** `jtag_axi` shim, one transaction, correctness only. ~1 ms per
round trip, so no throughput claim.
*Exit:* one decode step bit-exact against the on-target CPU golden model.

**Step 13 — PS/DMA, baremetal A53.** Vitis SDT flow, `xsdb`, UART capture.
*Exit:* multi-token, multi-context, bit-exact; device-side µs reported; **measured
DDR bandwidth under real load matches step 1**.

**Step 14 — Ethernet.** Raw Ethernet frames, small custom header, polled, no OS.
Extend `sa_proto.h` with an `ATTN` command; validate against `mock_server.c` on
x86 first.
*Exit:* end-to-end from the host with the GPU running the rest of the model;
report tok/s at ctx 8k / 16k / 32k against the table above.

**Step 15 — the result.** Perplexity from a run where attention genuinely
executed on the FPGA, then sweep `KEY_BITS`/`VAL_BITS`.
*Exit:* measured bytes, cycles and throughput move as `config.py` predicts, and
throughput is a linear function of the compression ratio.

### What can be done in parallel

Steps 2–4 (Python) are independent of step 1 and can run alongside it. Steps 5–9
are independent of each other once step 3 exists. Step 10 depends only on step 1.
The serial spine is **1 → 3 → 11 → 12 → 13 → 14 → 15**.

---

## System path

```
host GPU (torch)  ──►  numpy, Q8.16 packed 24-bit
        │                    │
        │              raw Ethernet over GbE (RJ45, TI DP83867 PHY, UG1267 p.44)
        │                    ▼
        │            ZCU104 PS — A53 baremetal, polled
        │                    │  AXI DMA MM2S
        │                    ▼
        │              PL: attn_top  ◄── 4× AXI-HP ──► KV cache in PS DDR4
        │                    │  AXI DMA S2MM
        ◄────────────────────┘
   W_o' (R⁻¹ folded in), residual, MLP, next layer
```

Reuse the protocol pattern in `fpga_work/systolic_zcu104/host/sa_proto.h`: a fixed
header plus a payload that is **exactly the beat stream the accelerator
consumes**, so the PS hands it to the DMA without repacking. Add an `ATTN` command
alongside `SA_CMD_GEMM`. Reuse `mock_server.c` — an x86 stand-in that unpacks
beats the way the RTL does — to validate the wire format and the host client
before board-side software exists. The 24-bit packer must be checked against the
RTL unpacker the way `quantize.pack` already is.

### Where the time goes

At the long contexts this plan targets, **DRAM dominates and the network is a
minor term.** Per layer, per token:

| ctx | cache/layer | DDR read | wire | RTT (raw Eth) | **ms/token** | **tok/s** |
|---|---|---|---|---|---|---|
| 8,192 | 3.4 MB | 232 µs | 123 µs | 35 µs | 4.27 | 234 |
| 16,384 | 6.8 MB | 463 µs | 123 µs | 35 µs | 7.97 | 125 |
| **32,768** | 13.6 MB | 927 µs | 123 µs | 35 µs | **15.39** | **65** |
| 65,536 | 27.3 MB | 1,854 µs | 123 µs | 35 µs | 30.22 | 33 |
| 131,072 | 54.5 MB | 3,708 µs | 123 µs | 35 µs | 59.88 | 17 |

Every layer re-reads its entire cache every token, so this is linear in context
and there is no way around it in this architecture.

Cache residency in the 2 GB PS DDR4: **218 MB at ctx 32k, 872 MB at ctx 131k** —
the model's full context fits, which is the point of compressing 4.92×.

**Network priority is therefore low.** Raw Ethernet frames with a small custom
header, polled, no OS on the A53, is still the right target — it takes RTT from
250 µs to 35 µs, worth ~17% at ctx 32k. Do it, but after the DDR path works. Keep
a TCP mode for the mock server and bring-up.

**The one lever with real leverage is bytes, not logic:**

| format | row | ports = ceil(row/16) | ctx 32k traffic | tok/s |
|---|---|---|---|---|
| 4b/4b | 68 B | **5 — more than the PS has** | 285 MB | 50 |
| **4b/2b** | **52 B** | **4** | **218 MB** | **65** |
| 3b/2b | 44 B | 3 | 185 MB | 77 |

On a DDR-bound build, **throughput is a direct linear function of the compression
ratio.** That makes the cache format the measured variable — which is exactly the
claim this project exists to support.

### Considered and rejected: USB as the data path

- **J96 (USB 3.0, 5 Gb/s):** UG1267 p.35 — the USB3320 PHY is **host-only mode**,
  confirmed by the MIC2544 VBUS current-limit switch and fault LED. It cannot
  enumerate on the PC. The same page's "connection to the host computer" sentence
  is boilerplate from other board guides and is wrong for this board.
- **J164 (FT4232H, JTAG + UARTs):** a real USB device port, but a JTAG/UART
  bridge. `jtag_axi` moves 32 bits per round trip at ~1 ms — **~82 s per token**.
  Even a custom streaming bridge at 30 Mb/s caps at ~11 tok/s.

Ethernet stays. JTAG keeps its existing role: first bring-up, proving one
transaction bit-exact, with no throughput claim attached.

### Considered and rejected: `W_o` on the FPGA

Folding `R⁻¹` into `W_o` is correct and already implemented (`ops/output.py:48`),
but it is not a saving of a matrix multiply — `W_o'` is 2048×2048, exactly
`W_o`'s shape. What it removes is the inverse butterfly: 32 vectors × 6 stages =
**32 cycles per layer** — which is why this plan folds and builds no inverse
hardware at all.

Moving `W_o` onto the PL to get that costs 8 MiB of fp16 weight streaming per
layer per token = **262,144 cycles** on a 256-bit port. Spending 262,144 cycles to
save 32 is **8,192× the wrong direction**, and it takes device traffic from 100%
cache to **91% weights** — exactly the outcome the RoPE split exists to avoid.

**Rule: fold `R⁻¹` into `W_o` wherever `W_o` already is; never move `W_o` to get
the fold.** The `folded` flag at `weights.py:39` keeps both paths available.

---

## Verification

**Numeric, before hardware**
```bash
python -m kernel.tests.run                        # existing 74 checks must still pass
python kernel/experiments/single_layer.py         # dense control: cosine 1.000000
```
Nothing in this plan touches a numeric path, so both must pass **unchanged**
throughout. If the dense control moves off cosine 1.000000, something that was
supposed to be cost modelling has leaked into the datapath — stop and find it.

**Saturation gate — before the wire format is locked in.** The 24-bit lane width
rests on a peak of 21.06 over 400 tokens of one passage, which is not enough
evidence on its own.
```bash
python -m kernel.experiments.wikitext_ppl              # wider text
```
Read `report.conversion.saturation_rate` and `max_abs_float` (`ConversionStats`).
**`saturation_rate` must be 0.** `FixedFormat.saturate=True` clamps rather than
wraps, so a non-zero rate is silent quality loss, not a crash.

**RTL**
```bash
kernel/scripts/regress.sh                         # per-block self-checking TBs
kernel/scripts/run_synth.sh                       # OOC synth: DSP/LUT/URAM/WNS
```
Every testbench compares against vectors from the numpy model, and the attention
TB compares against `attend_online` — never `attend_two_pass`.

**On hardware** — the exit criteria of steps 1 and 12–15 above.

The one that matters most is **step 1's DDR bandwidth measurement**: achieved
GB/s against the 13.0 GB/s the engine needs. The whole design is sized from it,
and it is the number most likely to disappoint. If it comes in low,
`PARALLEL_KV=1` is starved and every throughput figure in this document is
optimistic — which is exactly why it is measured before any attention RTL exists.

**The claim this supports, stated precisely.** Attention for Llama 3.2 1B ran on a
ZCU104 at long context with the KV cache compressed 4.92× and resident in DDR,
bit-exact against a numpy model that is itself exact against `LlamaAttention`,
with 100% of measured device traffic attributable to the cache and throughput a
linear function of the compression ratio. **Not** a claim that the model runs on
the board — it does not, and the weight analysis says why.

## Known limits to carry

1. Batch > 1 and `attention_mask` are refused by the kernel today
   (`adapters/torch_llama.py:108`). The PL block is decode-only by construction.
2. `d_head` must be a power of two — the FWHT has no other form. `d=64` also makes
   the `1/√d` scale an exact shift; a non-power-of-four `d` would put a multiplier
   back in the rotation.
3. Resource and timing numbers here are estimates, worth nothing until
   `run_synth.sh` and place-and-route have run.
4. DDR bandwidth is **measured**: 14.7 GB/s at 4 ports, burst 64, outstanding 2,
   reproducible to 0.25% across two runs and a power cycle
   (`results/step1_ddr_sweep_run2_medians.log`). It is a *read-only,
   four-stream, sequential* figure — the KV-scan pattern — and must not be
   quoted as general memcpy bandwidth. The write path (416 B/token/layer) is not
   included in it.
5. The grafted model writes zeros into `past_key_values`; anything inspecting the
   HF cache will be misled.
6. ~~The 21.06 dynamic-range peak behind the 24-bit lane width is 400 tokens of
   one passage.~~ **Closed by step 4**: 8,188 tokens, 402.7M elements, zero
   saturations, peak 24.957 against a ±128 rail. Still one corpus (WikiText-2)
   and one checkpoint, so a different domain could move the peak again — but
   the margin is 5.1×, and the gate is now a command that can be rerun rather
   than a paragraph.
7. Targets Llama 3.2 1B specifically: no QK-norm, `d=64`, 8 KV heads. A model with
   QK-norm or a different head dim changes the host half and, for a
   non-power-of-two `d`, the rotation itself.
8. Perplexity claims inherit the README's limits — 400 tokens of one in-domain
   passage separates a 2% effect from a 22% one and nothing finer.

## Reference: the numbers this plan is built on

Llama 3.2 1B, 16 layers, keys 4b / values 2b, ZCU104 (XCZU7EV: 504K logic cells,
1,728 DSP, 11 Mb BRAM, 27 Mb URAM, 2 GB PS DDR4-2400; the PL DDR4 SODIMM socket
J1 is **unpopulated** — UG1267 Table 1-2, so PS DDR4 is the only DRAM).

| quantity | value |
|---|---|
| attention weights, one layer, fp16 | 20.0 MiB — **4.2× all on-chip memory** |
| KV cache | 416 B/token/layer, 4.92× smaller than fp16 |
| engine appetite, `PARALLEL_KV=1` @ 250 MHz | 52 B/cycle = **13.0 GB/s** |
| DDR4-2400, **measured** (4 ports, burst 64, outstanding 2) | **14.7 GB/s** (77% of peak) — 13% headroom |
| one AXI-HP port @ 250 MHz | 4.0 GB/s → **4 ports required** (16.0 GB/s ceiling) |
| prefetch depth | **outstanding 2 per port** — deeper costs 19% |
| wire traffic | 245.8 kB/token, 16 exchanges, 1.97 ms |
| **throughput @ ctx 32,768** | **15.39 ms/token, 65 tok/s** |
| cache in DDR @ ctx 131,072 | 872 MB of 2,048 MB |

Weights dominate DDR traffic at every context below ~50k tokens, which is why the
projections stay on the host. Above ctx 8k the KV cache exceeds URAM, which is why
it lives in DDR. Those two facts together define the split.
