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

**Superseded by step 6:** the encoder's 78 cycles were an estimate of RTL that
did not exist yet. The synthesised blocks fold over the head dimension and cost
293. See step 6.

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

**Status: DONE.** `rtl/rot_norm.sv`, `rtl/rot_encode.sv`, `tb/tb_rot_norm.sv`,
`tb/tb_rot_encode.sv`. `run_synth.sh` and `synth_block.tcl` are now generic over
blocks, so `./scripts/run_synth.sh all` closes every block the datapath has.

`hw/vectors.py` gained the encoder stimulus: **560 vectors** — the 16 real
rotated k/v tensors, 512 random, and **32 constructed to sit exactly on a
decision boundary**. The tie vectors are the point. `_encode_plane` compares
with `>`, so a channel on a boundary belongs to the bin *below* it, and random
stimulus hits `x << 8 == norm * boundary` about once in 2³². They are built by
setting the last lane to the value that lands on a boundary and iterating to a
fixed point, because moving the lane moves the norm which moves the boundary.

Both benches were checked against a deliberately broken DUT before being
believed: `>=` in place of `>` fails on vector 528 lane 63 at *both* widths, and
half-up in place of half-to-even fails four norms out of 560. A bench that
cannot fail the mutation was not testing the rule.

The key plane (4 b, 15 boundaries) and the value plane (2 b, 3) run on the same
stimulus in the same bench, because the interesting bug is a boundary index
that is right at one width and off by one at the other. The boundaries are a
parameter checked against the emitted codebook, exactly as `SIGN_NEG` is against
the sign diagonal — a stale build fails there, not as a scattering of one-off
codes.

**Two blocks, two design decisions.**

*The norm is quantized before the thresholds are built from it.* `rot_norm` is
the only producer of the norm and everything downstream consumes what it
emitted. If the encoder thresholded against a wider norm than the cache row
carries, encoder and decoder would work from different numbers and a channel
near a boundary would decode into the wrong bin — a fraction of a percent of
accuracy, invisible to every test that does not look for it.

*Both blocks fold over the head dimension, 8 channels of 64 per cycle.* Phase A
runs once per token against a Phase B that runs once per **cached** token.
Sixty-four squarers and sixty-four comparator stacks would buy back cycles
nothing is waiting on, and take area from the score lanes, which are the thing
actually starved. The square root is one pipelined stage per output bit, so the
folding costs area, not throughput.

**Both first attempts missed 250 MHz, and synthesis is what said so.**

| | first attempt | fixed |
|---|---|---|
| `rot_norm` | **−0.990 ns** — lane mux + DSP multiply + 53-bit adder tree + the first sqrt subtract in one cycle | **+0.931 ns** — square → sum → mean as three registered stages |
| `rot_encode` | **−0.641 ns** — four 36-bit compares against mux-selected thresholds, chained | **+2.298 ns** — one registered level per bit |

Both fixes cost latency on a block nothing is waiting on. Neither was visible in
simulation; both were invisible to inspection.

Out-of-context, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| block | WNS | LUT | FF | DSP |
|---|---|---|---|---|
| `rot_fwht` | +3.309 ns | 12,385 | 11,361 | 0 |
| `rot_norm` | **+0.931 ns** | 2,366 | 2,773 | **16** |
| `rot_encode` | **+2.298 ns** | 2,493 | 2,483 | **14** |

The 16 DSPs are 8 squarers (24×24 takes two each); the 14 are the 15 threshold
multiplies, one of which is against a zero boundary and optimises away.

**The cost model moved, and this is the correction.** `EncoderConfig` claimed
one vector per cycle and an 8-cycle latency. The RTL retires one every **13**
(`D/LANES + KEY_BITS + 1`, the encoder being the slower of the two) with **31**
cycles of latency. So Phase A at ctx 4096 is **293 cycles, not 78** — 54 rotate
plus 239 encoder. Phase B is 32,786, so Phase B is still **112× Phase A** and
the conclusion ("optimise nothing but Phase B") is untouched, but the number
this document reported for step 2 was an estimate and this is the measurement
replacing it. `check_phase_b_is_the_whole_step_at_long_context` now asserts the
ratio, which is the claim, rather than the estimate.

The row layout closes the loop: `out_codes` is lane *i* at `[BITS*i +: BITS]`,
LSB-first, which is byte for byte what `_pack_codes` writes into a cache row,
and `check_the_encoder_goldens_agree_with_the_packed_cache_row` asserts the
bench's goldens are the codes `kv_rows.hex` actually contains. The two benches
cannot be verifying a row the other will never produce.

128 checks green; `./scripts/regress.sh` green on all four benches.

**Step 7 — `qtab_build.sv` + `score_lane.sv`.** The multiplier-free core: 64
parallel table gathers, six-level adder tree, one `k_norm` multiply, shift and
clamp. Ping-pong tables.
*Exit:* scores bit-exact against `CompressedAttention.scores` for a fixed cached
row; synth confirms ~3 DSP and ~3k LUT per lane.

**Status: DONE.** `rtl/qtab_build.sv`, `rtl/score_lane.sv`, `tb/tb_qtab_build.sv`,
`tb/tb_score_lane.sv`. `hw/vectors.py` gained the whole cache
(`cache_rows_h0..7.hex`), the codebook centroids as a format artifact
(`cb_centroids_{key,value}.hex`), the product tables (`qtab_table.hex`) and five
constructed rows (`score_edge_rows.hex`).

**2,080 scores bit-exact** — all 32 query heads against the 65 cached tokens
those scores were computed over — plus 32,768 table entries and 5 edge rows.
The golden is `CompressedAttention.scores` called by `hw/vectors.py::collect`
on the kernel's own attention object and the kernel's own cache: the same call
`forward` makes, not a reimplementation. The lane is handed the raw 52-byte row
and splits it itself, so `_pack_codes`'s bit layout is under test rather than
mirrored by a second decoder.

`tb_qtab_build` sweeps code *i* across **all D channels at once**, for every
*i*. A token reads D of the 1,024 entries, so driving real cache codes would
leave most of the table untested and would pass a builder that wrote the right
value to the wrong code index; the sweep is exhaustive and a transposed write
fails on the first head. Head *j+1* is built while head *j* is still being
gathered, and the overlap is *measured* (a flag set from `dut.busy`) rather than
inferred from relative durations — the first version of that check asserted
`out_valid` after a 32-cycle sweep of a 64-cycle build and failed the DUT for
the bench's arithmetic.

Both benches were checked against deliberately broken DUTs. `qtab_build`:
building into the bank being read, and a one-lane rotation of the channel index.
`score_lane`: round-half-up instead of floor (1,017 failures), the norm reaching
the multiply one stage early (1,204), and a perturbed value plane (2,083).

**The fourth mutation is the interesting one, because it survived.** Reading the
norm as *unsigned* agreed with all 2,080 real scores. `_unpack_word`
sign-extends, so the model multiplies by a signed 16-bit norm — but every norm
`isqrt` produces here is far below 2^15, so the rule is unobservable on real
data. `score_edge_rows.hex` puts the norm at 0, 1, 2^15−1, 2^15 and 2^16−1, the
last two negative once sign-extended, and the mutation now fails on two of the
five. A rule that no stimulus can violate is not being tested.

The last four heads are scanned with `out_ready` pulsed low at random — **239
stalled cycles**, asserted rather than assumed — and produce values identical to
the unstalled run. The first driver read the combinational `row_ready` in the
same delta it assigned `out_ready`, which raced the continuous assignment and
made the bench and the DUT disagree about which rows were accepted; that showed
up as isolated wrong scores in the stalled head only.

**The product table is LUTRAM, not URAM, and synthesis is what said so.** The
budget table above assumed 320 Kb of URAM. It cannot be: `score_lane` gathers
all 64 channels in the *same* cycle at 64 *different* addresses, so the table is
64 independent memories, not one wide one. The first attempt declared them as a
two-dimensional array and Vivado inferred **registers — 93,486 FF and zero
LUTRAM for a single lane**, which is 374k FF across four and does not fit the
part. Rewritten as D one-dimensional arrays with a scalar write address and an
asynchronous read, it infers `RAM32M16` and drops to 5,401 FF:

| | 2-D array | D separate arrays |
|---|---|---|
| FF | **93,486** | **5,401** |
| LUT | 26,659 | 3,461 |
| LUTRAM | **0** | **1,664** |

The width also moves: **43 b, not 40**. The widest gaussian centroid at
`KEY_BITS=4` is 89,465 — an 18-bit signed value, not the 16-bit one the estimate
assumed — so the table is 4 lanes x 64 x 16 x 43 b x 2 banks = **352 Kb of
distributed RAM**, 6,656 LUTRAM across the group.

Out-of-context, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| block | WNS | LUT | of which LUTRAM | FF | DSP |
|---|---|---|---|---|---|
| `qtab_build` | **+1.702 ns** | 3,461 | 1,664 | 5,401 | **16** |
| `score_lane` (lane only) | **+1.702 ns** | 862 | 0 | 1,591 | **2** |
| `score_lane` + its table | +1.702 ns | 6,387 | 1,664 | 7,011 | 18 |

**Two DSPs in the inner loop, which is the claim this step exists to check.**
The 49x16 norm multiply is split at bit 25 into two DSP-sized products and a
shifted add; synthesis absorbs both, along with the norm's pipeline register,
into the DSP48s. The estimate of "~3 DSP and ~3k LUT per lane" was for the inner
loop and holds at 2 DSP / 862 LUT; the **table build is 16 DSP and 3.5k LUT on
top of it**, which the estimate did not carry. Four lanes are therefore 8 DSP of
scoring and 64 DSP of table building — still 4% of the part, and still nothing
next to the 1,728 a dense unit would need, but it is 9x the figure this document
had.

Measured latency accept-to-score is **8 cycles**, not the 10 `AttentionConfig.
score_latency` carries. That constant covers S0..S10 — score *and* softmax — so
it is left alone until step 8 measures the other half. It is latency drained
once per scan either way and moves no conclusion.

131 checks green (three new in `tests/test_vectors.py`); `./scripts/regress.sh`
green on all six benches. No numeric module was touched.

**Step 8 — `softmax_online.sv` + `accum.sv`.** Speculative "no new max" commit,
flush-and-rescale on the rare path, 16×48 b elastic FIFO, 4-constant mux
accumulate. **exp LUT replicated ×4.**
*Exit:* bit-exact against `attend_online` — **not** `attend_two_pass` — at short
context, where rescales are frequent enough to exercise the flush path.

**Status: DONE.** `rtl/exp_lut.sv`, `rtl/softmax_online.sv`, `rtl/accum.sv`,
`tb/tb_exp_lut.sv`, `tb/tb_softmax_accum.sv`. `CompressedAttention.on_step` is a
new read-only observer on `attend_online` — the same `is not None` contract
`AttentionKernel.on_stage` uses — so `hw/vectors.py` taps the running `m`, `l`
and `acc` instead of reimplementing the recurrence in the vector generator.

**Bit-exact against `attend_online` on every token, not on the answer.** Two
stimulus sets x 4 lanes x 65 cached tokens: `m`, `l`, the probability, the
rescale factor and **all 64 accumulator channels** are compared after each
token. 39 rescale events, matching the count the model recorded lane for lane.
Plus 13,565 exp deltas.

**The exp LUT is 132 LUTs, not 8 BRAM36.** `ExpLut.__call__` collapses
table-then-shift into one gather over 4,353 entries; that is a numpy
optimisation (it was 28% of the model's runtime as two passes) and the budget
table above priced it as hardware — 68 Kb x 4 lanes, 272 Kb, 8 BRAM36. In
silicon the collapse is a pessimisation. `ExpLut.two_step` is the hardware form:
a 256 x 16 b distributed ROM, a 20x17 multiply and a barrel shift. Measured
**132 LUT, 16 FF, 1 DSP, zero BRAM** — 528 LUT across four lanes against 8
BRAM36 budgeted. `check_exp_lut_flat_matches_two_step` already asserted the two
agree; this is which one to build.

**Two stimulus sets, because the real scores do not exercise the rescale.** At
ctx 64 the four lanes take 19 new maxima and the factor is exactly unity in 13
of them, zero in the 4 first-token events, and something else **twice**.
Consecutive maxima differ by a few LSBs of Q16, so `exp(-delta)` rounds to 1.0
and `acc * factor >> 15` returns `acc` unchanged — the multiply runs and is
numerically inert. A mutation that rounds that rescale instead of flooring it
passed the entire real-data bench. Scaling the query by 1,024 (clipped to the
24-bit seam) spreads the maxima and yields **17 distinct factors** from 0 to
unity; the mutation then fails 130 checks. Same `attend_online`, same cached
rows — only the query is synthetic, and a query is exactly what the host is
free to send. `check_the_real_scores_do_not_exercise_the_rescale` pins the gap
so the extra set can be dropped if the golden data ever closes it.

**`accum`'s drain guard needed its own driver.** A SCALE must not be applied
while add terms are still in flight, or it scales a partial accumulator and
un-scaled terms land on top of it. But `softmax_online` drains its own pipe and
spends three more cycles on the factor before issuing one, so `accum` never
sees that case through this producer and the mutation removing the guard
passed. The bench now captures lane 0's op stream and **replays it into a
second `accum` at full rate**: the goldens are unchanged because the arithmetic
does not depend on the timing, a SCALE lands directly behind an ADD, and the
mutation fails 68 checks.

Also caught: `>=` for the new-maximum test (an infinite rescale loop — a
watchdog was added, because a deadlock that hangs iverilog reads as an
infrastructure problem rather than as the DUT being wrong), a double truncation
of the accumulator rescale, and two transposed value centroids.

**`rescale_cycles` was 1. It is 11, measured.** `tb_softmax_accum` counts the
bubbles directly — a score waiting in the input FIFO with no pop — over 39
events: drain the 3-stage exp pipe, 3 cycles to compute the factor from it,
issue the SCALE, wait while `accum` folds 64 channels through 8 multipliers.
**The conclusion is untouched and that is worth stating plainly:** new maxima
are logarithmically rare, so at ctx 32,768 this is 11 events and 121 cycles
against a 32,786-cycle scan — **0.37%**, against the 0.27% the 8-cycle estimate
gave. `score_latency` goes 10 → 11 (8 measured in `score_lane`, 3 in the exp
pipe); it drains once per scan and moves nothing.

**The input FIFO is 192 b wide, not 48.** The plan sized it counting only the
score. `accum` needs `v_idx` and `v_norm` for the same token whose probability
is emerging, and the only other way to get them is a second reader on the DDR
stream, so the entry is 48 + 128 + 16 bits. Depth is another matter: the FIFO
saturated at ctx 64 and that is an artifact, not a result — 19 rescales in 65
tokens is 11 cycles of stall for every 3.4 tokens, and no depth absorbs that.
It must cover **one** event, and one event is 11 cycles. Steady state cannot be
demonstrated at this context and is not claimed here.

Out-of-context, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| block | WNS | LUT | of which LUTRAM/SRL | FF | DSP | BRAM |
|---|---|---|---|---|---|---|
| `exp_lut` | **+2.204 ns** | 132 | 0 | 16 | 1 | **0** |
| `softmax_online` (with its exp) | **+0.377 ns** | 598 | 112 / 144 | 331 | 4 | 0 |
| `accum` | **+0.675 ns** | 7,427 | 0 | 2,487 | **17** | 0 |

`accum` is the largest block in the datapath so far, and the shape is the point:
17 DSPs, of which 5 are the inner loop (`v_norm * p`, then that against the four
value centroids) and the rest are the 8-wide rescale fold. The 7.4k LUTs are 64
saturating 32-bit adders and 64 4:1 muxes — the muxes are what a conventional
unit spends 64 multipliers on. `softmax_online` at +0.377 ns is the tightest
block in the design; the path is the 48x16 rescale of `l`, and it will want
watching when `attn_top` places it.

135 checks green (four new); `./scripts/regress.sh` green on all eight benches.
`config.py`'s `AttentionConfig` is the only non-RTL file whose numbers moved,
and it carries cycles, never a numeric result.

**Step 9 — `finalize.sv`.** One reciprocal divide per query head, `acc × recip
>> 16`.
*Exit:* end-of-scan output bit-exact against `_finalize`
(`ops/attention.py:553`).

**Status: DONE.** `rtl/finalize.sv`, `tb/tb_finalize.sv`. `collect` now keeps the
accumulator and denominator every head's scan actually ended on, for all 32
heads, and `check_the_finalize_goldens_are_the_kernels_own_two_lines` asserts
`reciprocal` + `_finalize` on those reproduces `out.online.hex` exactly — which
is the seam, so the block is checked against the value the FPGA returns and not
against an intermediate.

**44 rows bit-exact: 32 real heads and 12 constructed.** The constructed ones
are the point. Every head ends the same length of scan, so every `l` is near
`65 << 15` and every reciprocal lands between **1,008 and 1,012** — a four-value
band, all of them 10-bit. A divider wrong above ten bits, or on a zero or a
one, agrees with all 32. The extra rows put `l` at 0, 1, 2, 3, 2^15−1, 2^15,
2^16−1, 2^20, 2^31−1, 2^31, 2^32−1 and 2^33, which takes the reciprocal across
its whole range from 2^31 down to zero, with channel 0 of each a saturated
accumulator and channel 1 its negative so the widest product this block can
form is formed on every denominator.

`l = 0` is a real input, not an error case — a masked or padded position scores
nothing — and `reciprocal` answers zero rather than dividing. Removing that
special case fails 68 checks.

**The result is not clamped, deliberately.** `_finalize` does not clamp, so a
clamp here would make the block disagree with the model it is checked against.
The output is carried at 48 bits and `range_error` reports a channel the 24-bit
seam could not hold — the convention `rot_fwht` already uses. The bench checks
the flag against the golden's own width rather than merely observing it: 7 of
the 44 rows overflow the seam, all of them constructed, and **all 32 real heads
fit**, which is the property that matters.

Mutations caught: `>` for `>=` in the restoring compare (323), the `l = 0` case
(68), the `l >= 2^32` guard (68), and rounding instead of flooring the product
(1,259). One survived and is an equivalence rather than a gap: `>>` for `>>>` on
the product. The product needs at most `ACC_WIDTH + PROB_FRAC + 1` = 48 bits,
which is `OUT_BITS`, so the only bit where a logical and an arithmetic shift
differ is the one the truncation discards. It is written arithmetically anyway —
that is what the model does, and the equivalence is a property of one parameter
set, not of the operation.

**Slow on purpose: 50 cycles, 19 when the divide is skipped.** 32 of restoring
division plus 16 of folded multiply, once per head at the *end* of a T-cycle
scan. Widening either buys back cycles nothing is waiting on and takes DSPs from
the score lanes, which are the thing actually starved — the same argument that
sized `rot_norm`'s fold. At ctx 64 that is 77% of a scan and at ctx 8,192 it is
0.6%; overlapping it with the next head's scan is `attn_top`'s problem, not this
block's.

Out-of-context, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| block | WNS | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|
| `finalize` | **+1.443 ns** | 1,547 | — | 16 | 0 |

The 5,326 FFs are almost entirely two registers: the held accumulator (64 × 32)
and the output (64 × 48). Carrying the output at 48 bits when the seam is 24
costs about 1,500 FF, and it is what keeps the block bit-exact against an
unclamped model; narrowing it is a decision for `attn_top`, where the 24-bit
lane packing happens and `range_error` says whether it is safe.

137 checks green (two new); `./scripts/regress.sh` green on all nine benches.

### Phase 3 — memory and integration

**Step 10 — `kv_store_ddr.sv`.** 4× AXI-HP master, one per port; deep prefetch sized
from step 1's measured latency; separate non-blocking write path for the new
token's 8 rows. **The highest-risk block** — the only one whose correctness and
performance depend on a memory system rather than on vectors.
*Exit:* against a DDR behavioural model, sustains the step-1 bandwidth with the
score pipe never starved; row round-trip byte-exact against `quantize.pack`.

**Status: DONE.** `hw/ddr_layout.py`, `rtl/kv_plane_rd.sv`, `rtl/kv_store_ddr.sv`,
`rtl/kv_write.sv`, `tb/tb_kv_store.sv`, `tests/test_ddr_layout.py`.

**THE FLAT ROW LAYOUT ABOVE CANNOT FEED FOUR PORTS, AND THAT IS THIS STEP'S
FINDING.** This document specified the cache as packed 52-byte rows at a flat
stride, on the reasoning that a sequential address pattern is what a prefetcher
wants. It is — but a row is 52 B and one 128-bit HP port carries 16 B per cycle,
so four ports must run *at once* and each needs its **own** sequential stream. A
flat row layout contains exactly one stream.

Splitting the token range into quarters does not rescue it. The scan is causal,
so every token of the first quarter comes from port 0 alone, at 16 B/cycle
against the 52 B/cycle the score lane eats — and the only way port 0 keeps up is
if its whole quarter is already on chip. Four of those is **the entire cache
resident in BRAM**, which is the one thing this project exists not to do.

**So the cache is transposed into planes.** Split each row at its field
boundaries and chop any field longer than a beat into beats; at 4b/2b that is
four planes of 16, 16, 16 and 4 bytes, each held contiguously for every token.
Every port then reads one plane sequentially and the four together deliver one
row per cycle — three at a beat per token and the fourth at a beat per four.
The reassembly buffer is sized by *latency*, a few hundred bytes, not by
context.

The **row format is untouched**: these are the bytes `KVQuantizer.pack` wrote,
in its order, grouped differently in memory.
`check_the_reassembled_row_still_unpacks_to_the_cache` puts them back through
`unpack`, which knows nothing about either side.

*Fields first, not every 16 bytes.* Blind chunking is simpler and wrong for
**half** the `(key_bits, value_bits)` grid step 15 sweeps: `kb + vb = 3` gives a
28-byte row and a trailing plane of 12, and 12 neither divides a 16-byte beat
nor is a multiple of one, so that port would need a byte shift register. Every
field is `8*kb`, `8*vb` or 4 bytes — 0 or 8 mod 16, or 4 — so field-aligned
planes are always 16, 8 or 4 wide and all three divide the beat. All 64
combinations are swept, and 32 of them break under the blind split.

**One row per cycle in steady state, measured.** Against a behavioural DDR with
40 cycles of latency (worse than the board's) and two outstanding bursts per
port: **1,024 rows in 1,024 cycles** after the first, 44 starve cycles and all
of them at startup. That is 52 B/cycle = **13.0 GB/s at 250 MHz**, which is what
`PARALLEL_KV = 1` was sized against. The eight-head scan is byte-exact against
`cache_rows_h0..7.hex`, with and without random R stalls.

**Two bench bugs that looked exactly like DUT bugs, and are worth recording.**
The first slave model left `rvalid`/`rlast` asserted after a burst's last beat,
so the master counted an RLAST every cycle and its in-flight counter underflowed
to 213 — after which that port never issued another read and served stale FIFO
data forever. The symptom was wrong cache rows; the cause was a non-compliant
slave. There is now an assertion on the master's in-flight counter, which would
have localised it immediately.

The second is subtler and would have discredited the design: the model counted
down the *head* request's latency only, so a second outstanding AR did not begin
its round trip until the first finished. That makes outstanding-2 behave exactly
like outstanding-1 — the 8,458 MB/s row of the step-1 sweep rather than the
14,708 one — and the prefetcher measured 0.63 rows/cycle and looked like it hid
no latency at all. Every pending burst's latency now runs from when it was
accepted.

**The write path is a separate master.** 416 B per token per layer, 0.02% of the
traffic, but a write that stalled the read stream would stall the score pipe for
a rounding error's worth of bytes. One row is four single-beat writes, one per
plane, each naturally aligned to its own width — a narrow AXI transfer at
`AWSIZE=2` for the norms rather than a read-modify-write of a beat holding three
other tokens' norms. The bench round-trips **40 rows** (tokens 60–64 × 8 heads)
through WSTRB and back through the four read masters.

Mutations, all four caught after two rounds: a row that fires when *any* plane
has data (2,603), a write that ignores the byte lane (323), and the prefetch
depth moved to 1 or 4 (5 and 4 — the bench pins the measured depth, because a
build that quietly raised it would be 19% slower on the board and pass every
other check). Two needed the bench strengthened first: the round trip originally
wrote only token 64, whose norms land on byte lane 0 and exercise nothing, and
`plane_off(p)` is *equal* to `p*16` at 4b/2b — so a second DUT runs a
non-uniform 16/8/16/4 layout, where the two disagree, on synthetic stimulus.

Out-of-context, `xczu7ev-ffvc1156-2-e`, 250 MHz:

| block | WNS | LUT | of which LUTRAM | FF | DSP |
|---|---|---|---|---|---|
| `kv_store_ddr` (4 planes) | **+1.975 ns** | 3,795 | 2,368 | 967 | 0 |
| `kv_write` | **+0.245 ns** | 2,382 | 0 | 763 | 2 |

Zero DSP and zero BRAM in the read path: the 2,368 LUTRAMs are the four 256-beat
prefetch FIFOs, which is the whole on-chip cost of streaming a cache of any
length. `kv_write` at +0.245 ns is the tightest block in the design; the path is
the byte-lane placement, and it runs 32 times per token against a T-cycle scan,
so it can be pipelined freely if `attn_top` needs the slack.

**Carried to step 11.** The RTL is verified at 4b/2b plus one synthetic
non-uniform layout; the other 63 `(key_bits, value_bits)` combinations are
checked in Python only. Step 15 sweeps them and will need a second emitted image.

144 checks green (6 new); `./scripts/regress.sh` green on all ten benches.

**Step 11 — `attn_top.sv`.** Control FSM, 512-bit AXI4-Stream wrapper, 8 groups
of 6 vectors, k/v before q, raw-q hold (6 KiB) with just-in-time table build,
stream-through egress.
*Exit:* `tb_attn` bit-exact for multi-token, multi-context runs with
back-pressure; full `scripts/regress.sh` green; place-and-route meets 250 MHz
with positive WNS.

**Status: DONE.** `rtl/attn_ingress.sv`, `rtl/attn_top.sv`, `tb/tb_attn.sv`,
`scripts/impl_block.tcl`.

**The whole block, from the wire to the answer.** `tb_attn` hands the DUT
`ingress.hex` — 144 beats, 8 groups of 6 vectors — and checks the 32 heads it
returns against `out.online.hex`, bit for bit. Nothing between the two is given
to the hardware: the rotation, the norms, the codes, the cache row, the score,
the online softmax and the divide are all under test at once, which is the one
thing this level checks that ten block benches cannot. **4,170 cycles for
520 rows** at ctx 64, with 44 starve cycles and all of them at the first scan's
DDR round trip.

**Token 64 is BLANKED in the image, and that is the point.** `ddr_image.hex`
holds all 65 tokens because the kernel had already appended this one when
`collect` copied the cache. Left there, a DUT whose write path did nothing
would still answer correctly — the current token attends to itself and its row
would already be in memory. So the bench zeroes token 64's slot in every plane
of every head, and the only way it returns is through `kv_write`. The necessity
is asserted rather than assumed: with every write byte lane masked off, **all
2,048 channels change**.

**Back-pressure on both sides plus a stalling DDR is bit-identical.** The third
run stalls ingress, stalls egress and stalls R at random, and the result must
equal the clean run's exactly — 4,197 cycles against 4,170. A fourth run
repeats the step on the same instance, because every counter and the ingress
group counter have to come back to where they started or the second token of a
real decode is wrong in a way no single-step bench can see.

**FOUR BLOCKS NEEDED PIPELINING TO ROUTE, AND THEY ARE ALL THE SAME BUG.**
Every block met 250 MHz alone; `attn_top` first routed at **−0.550 ns**. The
failing path was the same shape four times over: a step counter fanning out
across a 64-word register file, through a wide mux, straight into a DSP.
Standalone that path has a whole die to spread across; with four lanes and
92k LUT placed, the mux output is a 1.0 ns route before the multiply even
starts.

| block | the path | fix | cost |
|---|---|---|---|
| `accum` rescale | `sgrp` → 8-of-64 mux → 48×16, two cascaded DSPs | select, then multiply | 9 cyc/event, was 8 |
| `finalize` fold | `mstep` → 4-of-64 mux → 32×32 | select, then multiply | one cycle per head |
| `accum` add | `a_term` → 64 4:1 muxes → 33-bit saturating add | mux, then add | one cycle of latency |
| `kv_write` | `pl` → address multiply → byte-lane placement | registered lane, `SETUP` state | one cycle per plane |
| `rot_norm` | `step` → 8-of-64 mux → squarers | select, then square | one cycle per vector |

None of it is arithmetic: the same operands, the same truncating shifts, the
same goldens, and every bench still green after each change. What moved is
`AttentionConfig.rescale_cycles`, **11 → 12**, which is measured by
`tb_softmax_accum` and not asserted anywhere — at ctx 32,768 that is 0.40% of a
scan against 0.37%. `plan.MD` had already said `kv_write` "can be pipelined
freely if `attn_top` needs the slack"; it needed it.

**One fix was mine and not a block's.** `head_base` is
`cache_base + grp*head_stride + p*plane_span` at 49 bits, and driving it
combinationally from `grp` put a 16-level carry chain between `grp` and the
plane readers' address adders. `grp` settles at the end of a group's egress and
the base is not read until that group's scan starts hundreds of cycles later,
so it is registered.

**Two bench bugs, both of which looked exactly like a deadlocked DUT.**
`attn_ingress` clears its beat counter on `start`, so a beat pushed in beside
that pulse is counted by the driver and dropped by the DUT — the step then
waits forever for a vector one beat short. And the back-pressure pattern was
written as `($time/4) % 5`: the clock period is 4 time units, so that modulus
is zero on *every* cycle and the egress stream was held off permanently rather
than randomly. The stalls are driven from a cycle counter now.

**Groups run one at a time, deliberately.** `PARALLEL_KV = 1`, so the eight are
sequential and only four query heads' softmax state is ever live. Overlapping
group *h+1*'s Phase A with group *h*'s scan is worth ~150 cycles against a
T-cycle scan — 0.5% at ctx 32k — and it would put ingress and the scan in the
same state, which is the one thing this block exists to keep readable. Not
done, and the cost is written down rather than hidden.

Place and route, `xczu7ev-ffvc1156-2-e`, 250 MHz, out of context:

| | WNS | LUT | of which LUTRAM | FF | DSP | BRAM/URAM |
|---|---|---|---|---|---|---|
| `attn_top` | **+0.053 ns** | 88,459 | 9,472 | 113,210 | 252 | **0** |

| instance | LUT | FF | DSP |
|---|---|---|---|
| `rot_fwht` | 12,408 | 11,361 | 0 |
| `accum` ×4 | 7,545 | 4,174 | 17 |
| `score_lane` ×4 (incl. `qtab_build`) | 5,378 | 7,009 | 18 |
| `kv_store_ddr` | 3,704 | 982 | 0 |
| `kv_write` | 2,608 | 815 | 0 |
| `rot_encode` k / v | 2,474 / 1,357 | 2,479 / 1,896 | 14 / 2 |
| `rot_norm` | 2,086 | 2,775 | 16 |
| `finalize` ×4 | 1,116 | 3,796 | 16 |
| `softmax_online` ×4 (incl. `exp_lut`) | 502 | 196 | 4 |
| `attn_ingress` | 10 | 1,542 | 0 |

**Zero BRAM and zero URAM in the whole block**, against the on-chip budget's
2 URAM of prefetch, 8 BRAM36 of exp LUT and 1.1 URAM of product tables. The
prefetch FIFOs and the product tables inferred as LUTRAM (9,472 of them) and
the exp LUT as a 252-LUT two-step ROM, which is what that block was written for.
The budget over-estimated because it sized a flat 4,353-entry table; the block
memories are simply unused and available for whatever the system build wants.

**254 DSPs, against the ~12 the datapath argument claims.** Both numbers are
right and the gap is worth stating: the "three DSPs in the inner loop" claim is
about the loop that runs once per CACHED token, and it holds — the gathers and
the 4:1 mux have no multipliers in them. The 252 are the things that run once
per *token* or once per *head*: building the product tables (16 per lane), the
reciprocal fold (16), the rescale fold (17), the norms (16) and the encoders.
They are amortised over a T-cycle scan and they are not what DDR is waiting on.

144 checks green (none new — this step added no Python; `config.py`'s
`rescale_cycles` moved and carries a cycle count, never a numeric result);
`./scripts/regress.sh` green on all eleven benches; `IMPL=1
./scripts/run_synth.sh attn_top` routes at +0.053 ns.

### Phase 4 — hardware bring-up

Mirror the order that already worked in `systolic_zcu104`.

**Step 12 — JTAG.** `jtag_axi` shim, one transaction, correctness only. ~1 ms per
round trip, so no throughput claim.
*Exit:* one decode step bit-exact against the on-target CPU golden model.

**Status: DONE, at 250 MHz.** `rtl/attn_ctrl.sv`, `tb/tb_attn_ctrl.sv`,
`scripts/create_bd_attn.tcl`, `scripts/build_attn_bit.tcl`,
`scripts/jtag_ps_init.sh`, `scripts/jtag_attn.tcl`.

**The block ran on the ZCU104 and returned all 2,048 channels bit-exact
against `out.online.hex`** — the numpy kernel's own answer for the same step,
which is the value the seam is defined by. Token 64's cache row reached DDR
only through the block's own AXI write master: the image is loaded with that
row zeroed, so a design that never wrote it would have scored a blank key
against itself.

```
### DDR round trip ok
### cache image loaded: 6784 words over 1696 beats, token 64 blanked
### token loaded and verified: 2304 words
### done after 1 polls, status 00000002
### starve 00000040  clips 00000000  overflows 00000000
### all 1536 words match out.online.hex
```

64 starve cycles, no score saturated, no accumulator channel wrapped, and no
channel escaped the 24-bit seam. (At 150 MHz the same run starves for 51
cycles: the DDR round trip is a fixed number of nanoseconds, so it is more
cycles at a faster clock. It is paid once per scan.)

**250 MHz IN THE FULL SYSTEM, AND IT TOOK SEVEN FIXES TO GET THERE.** The first
attempt missed by **−1.953 ns** with 49,557 failing endpoints, against
`attn_top`'s +0.053 ns out of context. A 150 MHz build was taken first and
passed on the board — correctness owes nothing to the clock — but a bitstream
that misses timing is worse than useless *for* correctness, because it produces
wrong answers indistinguishable from datapath bugs. So the timing was closed
rather than worked around.

**Out of context was the misleading measurement.** The core sat at +0.008 ns
with a whole die to spread across; every path that killed the system was one
the placer could hide there. The fixes were found by reading the routed report
after each build, never guessed:

| # | path | fix |
|---|---|---|
| 1 | FSM state → 6,144 `egr` flop enables (5.6 ns route, 0.226 ns logic) | the register was a second copy of `fin_vec`; index into it with a 12:1 mux instead |
| 2 | `grp` → 49-bit `grp*head_stride` → plane address adders | the group index only ever advances by one, so the base only ever advances by `head_stride`: one add, no multiplier |
| 3 | `accum`'s 4 terms → 64 channel muxes | `max_fanout` so synthesis replicates the registers |
| 4 | `softmax_online`'s 48×16 rescale of `l` | product registered, shift the cycle after |
| 5 | `a_w` (a DSP output) → four more DSPs | registered between them, so two multipliers are not in series |
| 6 | 64 channels each incrementing one overflow counter → 64 conditions on its clock enable | a registered popcount, folded in a cycle later |
| 7 | FSM state → 6,144 `q_vec` flop enables | a registered per-lane write enable, replicable |

**Fixes 4, 5 and the two products inside `accum` are one lesson.** `a_w *
centroid` is 32×20 and `acc * factor` is 48×16; both are wider than a DSP48 and
map to a cascaded PAIR, which does not reach 250 MHz without a register between
the multiply and the cascade adder. That register is the DSP's own MREG and
Vivado infers it **only if the RTL has one there**. Until they were added the
core sat at +0.008, +0.043, +0.018 — noise around a floor — because the limit
was inside the multiplier and every fix around it just exposed the next one.

**Fixes 1 and 7 are the other lesson**, and it is the same bug twice: a control
register decoded into the clock enable of thousands of flops. It never shows up
out of context and it is the first thing to break when the placer runs out of
room.

None of it is arithmetic. Same operands, same truncating shifts, same goldens,
twelve benches green after every single change. What moved is
`AttentionConfig.rescale_cycles`, **11 → 15**, measured by `tb_softmax_accum`
and asserted nowhere: at ctx 32,768 that is 0.50% of a scan against 0.40%.

**The margin is +0.006 ns, and that is thin enough to say out loud.** It meets
setup at the slow corner with hold at +0.010, and the board returns the right
2,048 numbers — but 0.15% of a period is not margin, it is a coin that landed
face up. A different placement seed or a hotter die could miss. Step 13 changes
this fabric anyway when the PS replaces the JTAG bridge, and if it comes back
negative the answer is the same as it was here: read the routed report, fix the
path it names.

**The shim is addressed memory, not a FIFO window.** 9,216 B in and 6,144 B
out, both readable back. A push-here/pop-there pair of magic addresses is
smaller and wrong for a bring-up: a `create_hw_axi_txn` that is retried or
typed twice shifts the whole stream by one word, and the failure looks like a
datapath that permutes channels. The host reads back every word it loaded
before starting anything, and `tb_attn_ctrl` checks that reading the output
window twice returns the same word — which a pop-on-read FIFO would fail and
nothing else would catch.

**DDR IS DEAD UNTIL SOMEBODY RUNS `psu_init`.** There is no FSBL and no A53
software in this design, so the DRAM controller is never brought up, JTAG
writes to it are acknowledged and discarded, and the block starts, issues its
first AR and sits at busy forever. Status `0x00000001` and no progress reads
exactly like a datapath hang. `jtag_ps_init.sh` does the bring-up over `xsdb`
and `jtag_attn.tcl` calls it — **after** programming the PL, because
`psu_ps_pl_isolation_removal` expects a configured PL and an earlier `rst
-system` cleared it, leaving Vivado with a device that had no debug core in it.
A two-pattern DDR round trip now runs before anything else, so the failure
names its cause in a second instead of surfacing minutes later as a hang.

**Four bring-up failures worth recording, none of them the datapath.**

| symptom | cause |
|---|---|
| identical LUT/FF/WNS to the digit after an RTL edit | `reset_run synth_1` does not reset the block design's **out-of-context child runs**; the top level re-links a stale DCP |
| `Protocol 'AXI4-Lite' does not support bursts` | `jtag_axi` is `PROTOCOL 2`; every transaction is one word. ~11,000 of them is seconds, so the script detects it and falls back rather than rebuilding |
| `Invalid hw_axi handle` after `psu_init` | `xsdb` on the JTAG chain invalidates Vivado's core handles; they must be re-enumerated with `refresh_hw_device` |
| `can't use "0x$st"` in `expr` | the hex prefix has to be inside a quoted string for `expr` to convert it |

The burst word order is **detected**, not assumed: a burst of known words is
read back with single-word reads, which have no ordering to get wrong. A
readback check cannot catch that class of error on its own, because writing and
reading with the same wrong transform agrees with itself.

Full system, `xczu7ev-ffvc1156-2-e`, routed:

| | WNS | WHS | FF |
|---|---|---|---|
| `attn_bd_wrapper` @ 250 MHz, seven fixes | **+0.006 ns** | +0.010 ns | 135,143 |
| `attn_bd_wrapper` @ 150 MHz, one fix | +0.552 ns | +0.010 ns | 127,771 |
| `attn_bd_wrapper` @ 250 MHz, as first built | **−1.953 ns** | +0.010 ns | 133,752 |

The last row is the design before any of this, kept because it is the
measurement that found the first bug. Utilisation at 250 MHz is 105k LUT
(45.7%), 252 DSP (14.6%), 2.5 BRAM and **zero URAM** — the on-chip memory
budget is essentially unspent.

**A build-flow trap worth its own line.** `reset_run synth_1` does not reset a
block design's out-of-context child runs, so an edit to `attn_top.sv` rebuilt
the PREVIOUS netlist and reported the previous utilisation to the digit and the
previous WNS to the picosecond. Nothing in the log says so. `build_attn_bit.tcl`
now resets every child run and prints the flop count, so a stale DCP shows up as
a number that did not move.

144 Python checks green (this step added none); `./scripts/regress.sh`
green on all twelve benches, `tb_attn_ctrl` being the twelfth; `scripts/jtag_attn.tcl` reports `JTAG_ATTN_PASSED` on the
board.

**Step 13 — PS/DMA, baremetal A53.** Vitis SDT flow, `xsdb`, UART capture.
*Exit:* multi-token, multi-context, bit-exact; device-side µs reported; **measured
DDR bandwidth under real load matches step 1**.

**Status: DONE, at 250 MHz.** `hw/board_vectors.py`, `tests/test_board_vectors.py`,
`sw/attn_main.c`, `scripts/create_bd_attn_ps.tcl`, `scripts/build_vitis_attn.py`,
`host/run_attn_a53.tcl`. Log in `results/step13_a53_f250.log`.

**Twelve decode steps, three contexts, every one of 18,432 channels bit-exact.**
Four steps at each of 64, 256 and 1,024 cached tokens, run back to back on one
loaded image. No clips, no accumulator overflow, no channel outside the 24-bit
seam, at any context.

**And the write path is what the multi-step run was for.** Each case loads an
image holding tokens `0..ctx0-1` and NOTHING else. Token `ctx0+n` therefore
scores against `n` rows that exist only because the block's own AXI write
master put them in DDR and the *next* scan's address generator found them
again. Step 12 ran one step and could not see that; `tb_attn` blanks one token
and cannot either. A write off by one plane or one token answers step 0
correctly and every step after it wrongly, and that is now checked twelve times
over three different plane geometries.

**A layout bug that capacity 65 was hiding, found before the board ran.**
`DdrLayout` gave each plane the pages its own width needed — at capacity 1,025
that is 20,480 B for the code planes and 8,192 for the norms. `attn_top` cannot
address it: its generator is ONE `plane_span` register walked as `cache_base +
p*plane_span`, because per-plane bases were the 49-bit multiply that missed
250 MHz by 1.953 ns (step 12, fix 2). Every plane now takes the widest plane's
span. It is invisible at capacity 65, where all four planes round to one page
anyway — which is why every vector set and every bench in this project had been
agreeing with a map the hardware could not have read.

### The scan approaches one row per cycle, and it does so from above

| context | rows | scan cycles | cyc/row | MB/s | starve |
|---|---|---|---|---|---|
| 65–68 | 2,128 | 8,243 | **3.87** | 3,356 | 3.1% |
| 257–260 | 8,272 | 15,188 | **1.83** | 7,080 | 1.8% |
| 1,025–1,028 | 32,848 | 41,008 | **1.24** | **10,413** | **0.7%** |

The per-group overhead is what the first column is made of: 191, 216 and 255
cycles per group against 66, 258 and 1,026 rows. It is roughly constant and the
rows are not, so it amortises — 74% of the scan at ctx 64 and 20% at ctx 1,024.
`PARALLEL_KV = 1` retires one row per cycle once the pipe is full, and 1.24 is
that number plus a fill and a drain paid eight times per step.

**So the ctx-64 bandwidth figure is a statement about pipeline fill, not about
DDR, and it would have been the only figure a single short case produced.**
That is why three contexts were run and why the longest is the one quoted.

### The DDR gate holds, and the starve fraction is what says so

**No run here can reproduce 14.7 GB/s, and one that claimed to would be
measuring the wrong thing.** Step 1 measured SUPPLY with four synthetic read
engines and no consumer. The engine's DEMAND is capped at 13.0 GB/s by the
datapath — one row per cycle, 52 B — and the measured 10.4 GB/s at ctx 1,024 is
that ceiling less the fill and drain above. Demand-limited, not supply-limited.

The falsifiable form of step 1's gate is the **starve fraction**, and across
these three contexts it **falls**: 3.1% → 1.8% → 0.7%. The absolute count
barely moves — 64, 70, 76 cycles per step — which reads as a fixed DDR round
trip paid once per group and then amortised over a longer scan.

> **CORRECTED BY STEP 14, AND THE CORRECTION MATTERS.** The paragraph that
> stood here concluded "a supply that could not keep up would stall a
> proportional share of every scan and the fraction would be flat or rising; it
> is neither." At **ctx 8,192 the fraction rises again, to 1.6%**, and the
> per-group starve goes from 9.6 cycles to **154.5** — 16× for an 8× context.
> Three points that happened to be monotone were read as a trend, and the trend
> was an artifact of never having measured past ctx 1,024. What the four points
> actually show is TWO components: a fixed per-group round trip, which does
> amortise, and a **per-row** term that does not — 0.0094 cyc/row at ctx 1,025
> against 0.0189 at ctx 8,193. The conclusion that the engine is fed survives;
> the reasoning that got there did not. See step 14.

### The transport is the step, and that is the measurement step 14 needs

| context | wall/step | block | transport | tok/s wall | tok/s if transport were free |
|---|---|---|---|---|---|
| 64 | 400 µs | 34 µs (8%) | 367 µs | 2,497 | 29,630 |
| 256 | 407 µs | 41 µs (10%) | 366 µs | 2,455 | 24,540 |
| 1,024 | 433 µs | **67 µs (15%)** | 366 µs | 2,309 | 14,981 |

`attn_main.c` reports load, run and read separately rather than one µs/token,
and this table is why: **85% of a decode step is the A53 moving 15 KB through a
32-bit AXI-Lite window one store at a time.** Reads cost 156 ns/word against
55 ns for writes — a write is posted and a read is a blocking round trip — so
the 6 KB read back costs 239 µs against 127 µs for the 9 KB written in.

**This is the measured case for a DMA, and it is deliberately not built yet.**
The plan's heading says "PS/DMA"; what was built is the PS as master with no
DMA, because `create_bd_attn.tcl` promised step 13 would replace the JTAG
master "and keep everything below it identical", and the shim is addressed
memory rather than a stream precisely so that swap would need no redesign. A
DMA path does need one. Building it first would have been sizing a fix against
a guess; the 366 µs above is the number to size it against, and it belongs to
step 14, where the Ethernet path needs a fast transport anyway.

### 250 MHz got EASIER when JTAG left the fabric

| | WNS | WHS | FF |
|---|---|---|---|
| `attn_bd_wrapper` @ 250 MHz, PS master (step 13) | **+0.082 ns** | +0.009 ns | 123,990 |
| `attn_bd_wrapper` @ 250 MHz, JTAG master (step 12) | +0.006 ns | +0.010 ns | 135,143 |

Step 12 closed at +0.006 ns and this plan called that "a coin that landed face
up" and said the answer, if step 13's fabric change made it negative, was to
read the routed report again. It did not: dropping `jtag_axi`, its smartconnect
port and JTAG's second path into HP3 removed 11,153 flops and returned **14×
the margin**. None of the seven step-12 fixes were reverted, and none of them
were wasted — the core still needs every one; it simply now has room.

**The clock is measured, not assumed.** The A53 divides `busy_cycles` by its own
wall-clock µs and reports the implied PL rate: **246–247 MHz** against a nominal
250, the 1.4% being the poll loop's overshoot inside an integer-µs measurement.
It is not there to calibrate anything. It is there so that a bitstream built at
150 MHz announces itself instead of quietly making every GB/s in the tables
above 40% too high.

**A note on what is NOT checked here.** The A53 reads back all 2,304 input
words before the first step of each case, so a dropped store on the AXI-Lite
path is caught where it happens rather than as a wrong answer three steps
later. It does not read them back on steps 1–3: by then the path has been
proven and the readback costs more than the step.

**Step 14 — Ethernet.** Raw Ethernet frames, small custom header, polled, no OS.
Extend `sa_proto.h` with an `ATTN` command; validate against `mock_server.c` on
x86 first.
*Exit:* end-to-end from the host with the GPU running the rest of the model;
report tok/s at ctx 8k / 16k / 32k against the table above.

**Step 13 handed this step its first requirement, measured.** 366 µs of every
433 µs decode step is the A53 walking 15 KB through a 32-bit AXI-Lite window,
against 67 µs of block. A GbE frame is 125 µs of wire time for 15 KB before
anything else, so a transport built on top of the current shim path would be
paying the 366 µs twice over. The DMA the system diagram already shows belongs
here, sized against that number rather than against a guess — and it is the
one change that needs the shim to grow a stream alongside its buffers.

**Status: PARTLY DONE. The transport exists and is tested; the board half of
the network does not.**

Done, and in the bitstream:

- **The DMA.** `attn_ctrl` grew a 512-bit AXI4-Stream pair selected by a MODE
  bit at 0x38; `create_bd_attn_dma.tcl` adds an `axi_dma` on **HPC0, its own
  coherent port**. Not a share of an HP port: all four carry a KV read master
  and the design is sized against them delivering 16.0 GB/s, so 15 KB a token
  would tax the one resource the architecture is short of. The buffers stay as
  MODE 0, because a bring-up that cannot re-read what it loaded cannot
  interpret what it gets out — and `attn_main.c` runs every case BOTH ways,
  checking each against the goldens, so the fast path agreeing with the slow
  one is not enough to pass.
- **`s_axis_tready` is gated on `busy`.** `attn_ingress` clears its beat
  counter on `start`, so a beat accepted before the step begins is swallowed
  and then discarded, leaving the stream one vector short — which surfaces as
  the LAST group hanging in S_RECV with every beat apparently delivered. The
  buffered path cannot reach it, because its feeder is armed by `start`
  itself. Software ordering would also fix it and would be a rule nothing
  enforces.
- **The wire format**, `host/attn_proto.h` and `host/attn_proto.py`. The
  payload is EXACTLY the beat stream the core consumes, so the A53 hands
  received bytes to the DMA untouched. The host owns the cache map —
  `cache_base`, `head_stride`, `plane_span` travel with every step and the
  board keeps no copy, because `hw/ddr_layout.py` computed the layout and also
  built the image.
- **`tests/test_proto.py` compiles the header and asks the compiler for the
  offsets**, rather than reading them off the source. Two definitions of one
  format is the bug that survives every unit test on both sides. It caught
  `uint64_t cache_base` twice — padding inserted before the field, then
  trailing padding because `sizeof` must be a multiple of the alignment — so
  the address is split into lo/hi words the way `attn_ctrl`'s own register map
  splits it.
- **`host/attn_mock.py` and `host/attn_client.py`.** The mock answers from
  `board_vectors`' table and does NOT recompute attention: a second golden
  model would prove nothing and would drift. A mismatch there means the
  transport is wrong; it cannot mean attention is.
- **ctx 8,192 added to the board cases** — the first row of the table below,
  and the regime the headline numbers are quoted at.

Not done:

- **The A53's Ethernet server.** Raw frames on the GEM, polled, no OS. The
  fragmentation is specified and the host side is written and tested against
  the mock; the board side is not.
- **The host integration** — `adapters/torch_llama.py` routing attention to a
  client instead of to numpy, with the GPU running the rest of the model. That
  is what the exit criterion's tok/s comes from, so **step 14 is not closed and
  no end-to-end throughput is claimed yet.**

### On the board: 32 steps, four contexts, both paths, all bit-exact

`results/step14_dma_f250.log`. Every case runs twice — once through the
AXI-Lite buffers, once through the DMA — and both are checked against the
goldens, so the fast path agreeing with the slow one is not what passes it.
**This is also the first hardware run of the normalised divide**, so every
golden in it is new.

**The DMA removes the transport, and the shorter the context the more it
matters.**

| ctx | AXI-Lite µs/step | DMA µs/step | speedup | transport, AXI-Lite → DMA |
|---|---|---|---|---|
| 64 | 400 | **30** | 13.3× | 367 → 3 µs |
| 256 | 408 | **36** | 11.2× | 367 → 3 µs |
| 1,024 | 433 | **64** | 6.8× | 367 → 4.5 µs |
| 8,192 | 699 | **330** | 2.1× | 367 → 5 µs |

Step 13's 366 µs is now ~5 µs and does not grow with context, because the
transfer overlaps the scan instead of bracketing it — `read` is 0 µs in every
DMA row, and that is not a missing measurement but a transfer that finished
inside the run. At ctx 64 the step is now 30 µs against a 27 µs block: the
shim has stopped being the measurement.

**The scan keeps converging on one row per cycle: 3.86 → 1.85 → 1.25 → 1.17.**
At ctx 8,192 the engine sustains **11.1 GB/s**, 85% of the 13.0 GB/s its
datapath can consume, and the remaining 15% is the per-group fill plus starve.

**Where that leaves the headline number, honestly.** At ctx 8,192 one layer's
attention is **325 µs** on the DMA path. Over Llama 3.2 1B's 16 layers that is
**5.20 ms/token, or 192 tok/s** — against the **234 tok/s** the table below
predicts. The 18% gap is not mysterious and is not measurement noise: the table
assumes the scan retires one row per cycle at the full 14.7 GB/s, and the
hardware retires 1.17 rows-worth of cycles per row at 11.1 GB/s.

**And the starve number is the one to carry forward.** Per group it is 8.0,
8.8, 9.6 and then **154.5** cycles at ctx 8,192. As a share of the scan it
falls, falls, falls and then rises — 3.1%, 1.8%, 0.7%, **1.6%**. Two
components, not one: a fixed round trip per group, and a per-row term that
roughly doubled between ctx 1,024 and 8,192 (0.0094 → 0.0189 cyc/row).

> **MEASURED, AND IT PLATEAUS.** The guess written here was "extrapolated to
> ctx 32,768 it is 3–4% of the scan". It is **1.5%**. The per-row term stops
> growing: **0.0202, 0.0184, 0.0183 cyc/row** at ctx 8k, 16k and 32k. See the
> probe below. Two guesses about this number in two days, both wrong in
> opposite directions, which is the argument for the probe existing at all.

### The long-context probe: DDR keeps up at 32k

`results/step14_longctx_probe.log`. Answering this bit-exactly needs a 16.9 MB
image in the ELF and an O(T²) numpy prefill, and neither is needed to measure
an ADDRESS PATTERN — so the A53 fills DDR itself and **nothing here is compared
against a golden**. The answers are wrong by construction and are never looked
at.

| ctx | cyc/row | MB/s | starve/group | **starve cyc/row** | % of scan |
|---|---|---|---|---|---|
| 8,193 | 1.16 | 11,116 | 165.7 | **0.0202** | 1.7% |
| 16,385 | 1.15 | 11,273 | 302.7 | **0.0185** | 1.6% |
| 32,769 | **1.14** | **11,331** | 601.1 | **0.0183** | **1.6%** |

**The per-row starve is flat and the per-group figure simply tracks the row
count** — 165.7, 302.7, 601.1 for 8k, 16k, 32k, which is linear. So above ctx
8k the fixed round trip is gone into the noise and starving is one steady-state
rate: about one cycle in fifty-five. It does not compound. **`PARALLEL_KV = 1`
at 4b/2b is fed at every context this design targets**, which is what step 1's
gate was for, now measured at the far end rather than extrapolated to it.

**The control is what makes the other two rows worth reading.** `probe8192` runs
the same geometry as the bit-exact ctx-8,192 case and recomputes its span and
stride from the same rule `ddr_layout.py` uses:

| | cyc/row | MB/s | starve/step | span / stride |
|---|---|---|---|---|
| probe8192, PRNG cache | 1.16 | 11,116 | 1,326 | 135,168 / 540,672 |
| ctx8192, real cache, checked | 1.17 | 11,090 | 1,279 | 135,168 / 540,672 |

Within 1% on every column. The synthetic image is representative, so the 16k
and 32k rows are about the machine and not about the filler.

### The throughput table was optimistic, and by a nameable amount

`cyc/row` improves all the way out — 3.87, 1.85, 1.25, 1.16, 1.15, **1.14** —
so the scan really is converging on the one row per cycle `kv_store_ddr` was
built for. But it converges from above, and the table below divides the cache
by **14.7 GB/s**, which is the *supply* step 1 measured with four synthetic
engines and no consumer.

**The engine's ceiling is demand, not supply**: one row per cycle is
`52 B × 250 MHz = 13.0 GB/s`, and what it achieves is that divided by
`cyc/row`. That model is now predictive to 1%:

| ctx | 13.0 / cyc-per-row | measured |
|---|---|---|
| 8,192 | 11.2 GB/s | 11.1 |
| 16,384 | 11.3 GB/s | 11.3 |
| 32,768 | 11.4 GB/s | 11.3 |

So the corrected figures for Llama 3.2 1B's 16 layers, from the measurement
rather than from the bandwidth:

| ctx | µs/layer/token | ms/token | **tok/s measured** | tok/s predicted below | optimism |
|---|---|---|---|---|---|
| 8,192 | 307 | 4.91 | **204** | 234 | 1.15× |
| 16,384 | 605 | 9.67 | **103** | 125 | 1.21× |
| **32,768** | **1,203** | **19.25** | **52** | 65 | **1.25×** |

The table below is left as written, because it is what the design was sized
from and the sizing decisions it drove — four ports, `PARALLEL_KV = 1`, 4b/2b —
are all still correct. But **no throughput number in it should be quoted
without this correction**, and the 32k row in particular is **52 tok/s, not
65**.

**A measurement artifact worth naming so nobody chases it.** The implied PL
clock reads 226–249 MHz and the low readings are all in DMA mode at short
context. It is integer-µs truncation: at ctx 64 the DMA step is 27 µs, so ±1 µs
is ±3.7%. The ctx-8,192 rows, where the step is 325 µs, read 248–249 MHz. The
clock did not move; the ruler got shorter.

**The DMA costs a little starve, consistently.** 292 against 256 at ctx 64, 386
against 308 at ctx 1,024, 5,058 against 4,943 at ctx 8,192 — a few percent more
in every case. HPC0 is a separate port but not a separate DRAM, so the token
transfer does contend with the scan for the controller. It is small enough to
note and not to act on.

Full system, `xczu7ev-ffvc1156-2-e`, routed:

| | WNS | WHS | FF |
|---|---|---|---|
| `attn_bd_wrapper` @ 250 MHz, PS + DMA + normalised divide | **+0.161 ns** | +0.010 ns | 143,181 |
| `attn_bd_wrapper` @ 250 MHz, PS master (step 13) | +0.082 ns | +0.009 ns | 123,990 |
| `attn_bd_wrapper` @ 250 MHz, JTAG master (step 12) | +0.006 ns | +0.010 ns | 135,143 |

The margin went UP while the design grew by a DMA, a smartconnect, a coherent
port and two pipeline stages — because taking the priority encoder out of the
divide loop bought more than all of that cost.

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
9. **THE FINAL DIVIDE LOSES PRECISION LINEARLY IN CONTEXT, AND EVERY CONTEXT
   THIS PROJECT HAS TESTED IS A LUCKY ONE.** `finalize` computes
   `recip = floor(2**31 / l)` in Q16 (`ops/attention.py:170`,
   `rtl/finalize.sv`), and the online denominator measures
   `l = T * 2**15` to within 0.2% across six contexts — every score is near
   the max, so every `p` is near unity. So `recip = 2**16 / T`, and the
   divide's precision is `log2(recip)` bits:

   | ctx | 1,024 | 8,192 | 16,384 | 20,000 | 24,576 | 32,768 | 65,536 | 131,072 |
   |---|---|---|---|---|---|---|---|---|
   | `recip` | 64 | 8 | 4 | 3 | 2 | 2 | 1 | **0** |
   | rel err | 0.2% | 0.2% | 0.2% | 8.6% | **25.2%** | 0.2% | 0.2% | **100%** |

   The error is the fractional part of `2**31/l`, so it is a LOTTERY: at powers
   of two `l` lands just above an integer and the error is 0.2%, and at ctx
   24,576 it is 25%. The tested contexts are 32, 64, 256, 1,024 and 4,096 —
   all powers of two, all winning tickets. The guaranteed bound is `T/2**16`,
   which is **above 25% for any context past ~16k**, and at 131,072 `l_too_big`
   fires and every output channel is zero.

   **Bit-exactness cannot catch this.** `ops/attention.py` and `finalize.sv`
   perform the same integer divide, so the model and the hardware are wrong
   together and every check in the project passes. Step 13 passed at ctx 1,024
   because `recip` was still 64 there.

   This bites inside the target range: **ctx 32,768 is the bolded row of the
   throughput table above and ctx 131,072 is its last row.** No throughput or
   perplexity number may be quoted at a context past ~16k until the divide is
   normalised — shift `l` to its leading one, divide, and carry the shift into
   the final right-shift, which holds ~31 bits at every context. That is a
   deliberate numeric-path change and it invalidates every golden, so it is
   its own step and not a patch inside another one.

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
