# `kernel/` — a compressed KV cache, in numpy and on an FPGA

One attention block, written twice and checked against itself: a numpy model
you can drop into a real LLM, and RTL that runs the same arithmetic on a
ZCU104. The two agree bit for bit.

The point of the design is that a cached key is never decompressed. It is
stored as `norm * centroid[code]` in a rotated space, and the score is computed
straight off the codes:

```
<q, k>  =  <R q, R k>                     R is orthonormal
        =  <rq, norm * centroid[code]>
```

The query is rotated once per step. No cached token is ever rotated, and no
dense key or value is ever built. The same trick works on the value side, so
the output stays in the rotated space and `R⁻¹` is folded into `W_o` once,
offline.

## Where it runs

Llama 3.2 1B, all sixteen layers' attention on the board, one token at a time,
with every cache row written by the FPGA's own AXI write master. Nothing is
preloaded and there is no prefill anywhere.

| | |
|---|---|
| decode steps checked against numpy | **4,096 of 4,096 bit-exact** |
| perplexity, fp32 → 4b/2b on the board (4,096 tokens) | **9.074 → 10.640 (1.173×)** |
| cache size | **4.92× smaller** |
| decode step, on the device | 22–26 µs |
| decode step, over the wire (UDP) | 1.5 ms |
| scan rate at context 32,768 | 1.14 cycles/row, 11.3 GB/s |
| timing, whole system at 250 MHz | WNS +0.161 ns, 143,181 FF |

So 4-bit keys and 2-bit values cost about 17% of perplexity and save about 5×
of cache. That is the trade the whole design exists to make, and it is measured
end to end with the hardware in the loop rather than modelled.

`plan.MD` is the engineering log: what was built, in what order, what broke,
and what is still open.

## The two halves, and the seam between them

Attention is split in one place, right after RoPE:

```
x ─┬─ W_q ─┐
   ├─ W_k ─┼─ projections                        HOST
   └─ W_v ─┘
          │
     fp32 → Q8.16                 ← ops/convert.py, the one cast in the package
          │
        RoPE                      ← position enters here, before the cache
          │
════════════ the seam: 24-bit lanes, 3 beats a vector ════════════
          │
     R (randomized Hadamard)      ← one round, on q, k and v
          │
   quantize K,V → DDR cache       ← never rotated or decompressed again        FPGA
          │
     score on codes → online softmax → accumulate
          │
════════════ back across the seam: Q16, merge_heads order ════════════
          │
        W_o'                      ← R⁻¹ already folded in                      HOST
```

Everything above the seam is O(1) in context. Everything below it is the KV
cache, which is the thing being measured — so 100% of the FPGA's DDR traffic is
cache traffic, and no byte of it is weight streaming.

## Running it

The numpy side needs only numpy:

```bash
.venv/bin/python -m kernel.tests.run          # 170 checks, no pytest needed
.venv/bin/python -m kernel.demo 512           # cost of a decode step at context 512
```

The RTL benches are self-checking and run under Icarus:

```bash
./scripts/regress.sh                          # all 12 benches
./scripts/regress.sh finalize                 # just the ones matching "finalize"
```

Icarus lives in the `ubuntu-work` distrobox rather than on the RHEL host, and
the script enters it for you. Vivado and Vitis are the other way round, host
only. Set `KERNEL_NO_DISTROBOX=1` if you are already inside the container.

The benches read their goldens out of `tb/vectors/`, which comes from
`python -m kernel.hw.vectors`. Regenerating those files and re-running the
benches is what checks the model and the RTL against each other.

The board goes through one wrapper, which picks the working directory and the
interpreter, reprograms the board if it does not answer a ping, captures the
UART to `/tmp/attn.log`, and prints the board's own lines at the end:

```bash
./scripts/attn infer --tokens 8 --trace    # all 16 layers, live, over UDP
./scripts/attn ppl   --tokens 4096         # perplexity, board against baseline
./scripts/attn diff                        # board's cache bytes vs the host's
./scripts/attn infer --cpu-only            # the control: no board, no sudo
```

Add `--no-verify` to skip the per-step comparison against numpy. That check is
O(context) and dominates a long run — at context 4,096 it is 675 ms a
layer-step against 9 ms for the ingress the board actually needs. Verify while
you are establishing that a configuration is exact; turn it off to measure.

Against the real model on the host alone (needs `transformers`):

```bash
.venv/bin/python kernel/experiments/single_layer.py     # one layer vs LlamaAttention
.venv/bin/python kernel/experiments/full_model.py 400   # every layer, perplexity
.venv/bin/python -m kernel.experiments.ablate_bits 400  # keys vs values
.venv/bin/python kernel/experiments/generate.py         # actual text
```

## How the package is put together

The rule that shapes everything: **ops produce values and a trace; the hardware
model replays the trace to produce cycles.** Timing watches, it never takes
part. No op can see a `HardwareConfig`, so the hardware model cannot change a
number — `tests/test_kernel.py::check_hardware_config_cannot_change_a_value`
asserts exactly that.

What that buys: with timing off this is an ordinary attention module you can
graft into a model, and with timing on it is an accelerator model. One
implementation, not two kept in step by hand.

| module | what it is for |
|---|---|
| `kernel.py` | composition — the only file that knows the pipeline order |
| `weights.py` | the four projections, with the offline folds applied |
| `ops/` | pure functions: project, rope, rotate, quantize, attention, output |
| `cache/` | the `KVCache` interface, dense fp16 and compressed |
| `numerics/` | `fp.py` (fp16×fp16→fp32) and `fixed.py` (the only narrowing) |
| `hw/` | cycles and bytes replayed from a trace, and the golden vectors |
| `host/` | talking to the board: UDP, raw Ethernet, JTAG |
| `rtl/`, `tb/` | the SystemVerilog and its benches |
| `sw/` | what runs on the board's A53 |
| `trace.py` | the seam between values and timing |
| `config.py` | one validated dataclass tree |

Three rules the package holds itself to:

1. **One float→fixed cast**, in `ops/convert.py`. A cast added wherever it was
   convenient is how two implementations of the same arithmetic end up
   disagreeing on inputs nobody tested.
2. **One narrowing module.** Intermediates stay int64 and grow. A value only
   gets narrower at a named `Q.rshift`, `Q.clamp` or `Q.from_float`.
3. **Prefill and decode are the same code.** `forward` takes
   `(tokens, hidden)`; one row is a decode step. There is no separate
   steady-state path.

## No calibration files

Rotating by a randomized Hadamard makes the channels close to Gaussian — that
is what the rotation is *for* — so the best scalar codebook afterwards is the
Lloyd–Max quantizer for `N(0,1)`, which depends on the bit-width and nothing
else. `Codebook.gaussian(bits)` works it out at construction.

That removes a whole class of infrastructure: no calibration pass, no exported
ROM, no JSON the hardware and the model have to be kept in sync on, and no way
to write a cache under one calibration and read it under another. The result is
checked against the published Max & Lloyd optima
(`tests/test_codebook.py::check_matches_published_lloyd_max`).

## One rotation round, not two

The default is one round, and that was a measured decision rather than a
guess. Two rounds looks better on paper: a single `HD` applied to a one-hot
vector gives `±1/√d` in every channel, which is a two-point distribution with
excess kurtosis of exactly −2.

| input | 1 round | 2 rounds |
|---|---|---|
| one-hot (pathological) | −2.00 | −0.25 |
| 4 outlier channels ×30 | −0.87 | −0.14 |
| student-t, df = 2.5 | −0.26 | −0.08 |

All of that is true, and none of it predicts quality. Measured end to end on
WikiText-2 (12 windows, paired, TinyLlama at 1024 context, 4b keys / 2b
values), one round beats two by **1.3% perplexity, at 2.8 standard errors**.
Three rounds is indistinguishable from two.

The likely reason is truncation. The `1/√d` scale is applied *inside* each
round, so a second round rounds the value off a second time, and that costs
more than the better distribution gives back. Kurtosis says how Gaussian the
channels look, not what the quantizer can actually use. A proxy that resembles
the objective is not the objective, and the only way to find that out was to
measure the thing we actually cared about.

One round is also half the hardware. See `experiments/ablate.py`.

## What the widths cost

TinyLlama-1.1B, all 22 layers grafted, perplexity over 400 tokens
(`experiments/full_model.py`). These are host-side numbers and predate the
board.

| configuration | ppl | vs base | B/tok/layer | vs fp16 |
|---|---|---|---|---|
| baseline (real fp32 attention) | 11.363 | — | 1024 | 1.00× |
| **kernel, dense (control)** | **11.363** | **1.000×** | 1024 | 1.00× |
| key 6b / value 6b | 11.510 | 1.013× | 400 | 2.56× |
| key 4b / value 4b | 11.589 | 1.020× | 272 | 3.76× |
| **key 4b / value 2b** | **12.112** | **1.066×** | **208** | **4.92×** |
| key 3b / value 2b | 13.902 | 1.223× | 176 | 5.82× |
| key 2b / value 2b | 34.725 | 3.056× | 144 | 7.11× |

The control row is the one that matters. With quantisation off the kernel
reproduces the real model's perplexity exactly, so the projections, the cast,
RoPE, the rotation, the codes path and the `W_o` fold are all correct, and
every row below it is attributable to compression alone. At the single-layer
level the same control matches `LlamaAttention` to 4.4e-4 relative, cosine
1.000000 (`experiments/single_layer.py`).

### Keys are expensive; values are nearly free

Sweeping one width with the other fixed at 4 bits (`experiments/ablate_bits.py`):

| bits | keys swept | values swept |
|---|---|---|
| 6 | 1.003× | 1.039× |
| 5 | 1.016× | 1.027× |
| 4 | 1.020× | 1.020× |
| 3 | **1.141×** | 1.044× |
| 2 | **2.351×** | **1.066×** |

Keys fall off a cliff below 4 bits: 3 bits costs 14%, 2 bits destroys the
model. Values degrade gracefully — going all the way down to 2 bits costs 6.6%.

So the two planes should not be the same width, and the value plane is where
the bits should come from first. `key_bits=4, value_bits=2` is the best point
measured. A symmetric format of about the same size (3b/2b, 5.82×) costs 22.3%,
which is more than three times the damage for 18% more compression.

## Grafting into a real model

```python
from kernel.adapters.torch_llama import graft
from kernel import QuantConfig

original = graft(model, layers=[10], quant=QuantConfig(key_bits=4, value_bits=2))
```

torch is imported lazily and is not a package dependency.

Note that `QuantConfig` defaults to `key_bits=3`, while the bitstream is built
for 4. Passing the widths explicitly is worth the keystrokes: a mismatch here
cost a full day once, because it produces a cache the hardware writes at one
row length and the host computes at another, and almost every symptom looks
like a datapath fault. `FpgaLayers` now checks the row width against the
bitstream before the first token.

## Known limits — carry these, do not drop them

1. **Batch > 1 and `attention_mask` are refused**, not tolerated. The kernel is
   causal by construction, which covers decoding but not padded batches or
   custom masks. The FPGA block is decode-only by design.
2. **`head_dim` must be a power of two.** The rotation is a Walsh–Hadamard
   butterfly and has no other form. This is an architectural limit, not an
   unimplemented case. `d = 64` also makes the `1/√d` scale an exact shift.
3. **The adapter is verified against transformers 5.16.1 and nothing else.**
   That contract moves — 5.x nests `rope_theta` inside `rope_parameters`, and
   the layer passes `position_ids` rather than `cache_position`. Re-run
   `experiments/single_layer.py` after any version bump; its dense row must
   come back at cosine 1.000000.
4. **Prefill and decode are not bit-identical**, they agree to about 2e-7
   relative. The fp32 GEMM blocks differently at different batch sizes, so its
   accumulation order changes. Everything below the fixed-point boundary is
   exact; this is fp32 reassociation and nothing else.
5. **Online and two-pass softmax are not bit-identical either**, and must not
   be asserted to be. The online form truncates the accumulator once per
   rescale. The hardware runs online; `forward` runs two-pass.
6. **The grafted model writes zeros into `past_key_values`.** That object is
   bookkeeping only and the kernel never reads it, but anything inspecting it
   will be misled.
7. **Bit-exactness is established up to context 4,096**, and only up to 256
   with live model activations. The datapath is checked at contexts 64, 256,
   1,024 and 8,192 on synthetic vectors. Nothing has been checked at 32,768.
8. **Perplexity is one corpus and one checkpoint**, measured over a single
   contiguous passage of WikiText-2 rather than sampled windows. It also mixes
   "how good is 4b/2b" with "how much context was available", because the
   context grows through the run.
9. **The cycle model in `hw/` is analytical.** It is a schedule model with a
   stated dataflow, not a number from a synthesis tool. Its value is in showing
   which bound binds. The board numbers at the top of this file are measured;
   these are not.
