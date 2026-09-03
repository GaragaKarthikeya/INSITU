# `kernel/` — a compressed-KV attention block, from scratch

One attention block as a function: token embeddings in, token embeddings out.
Written to be dropped into a real LLM, and instrumented so the same code also
answers what it would cost in hardware.

```python
from kernel import AttentionKernel, KernelConfig, ModelConfig

k = AttentionKernel(KernelConfig(model=ModelConfig.tinyllama()))
y, report = k.forward(x)          # (tokens, hidden) -> (tokens, hidden)
```

Nothing here is shared with `software/`, `hls/` or `chisel/`. It is a separate,
self-contained package with numpy as its only hard dependency.

## The idea

A cached key is stored as `norm * centroid[code]` in a rotated domain. The
conventional thing is to rebuild the dense vector and then dot it with the
query. This never does:

```
<q, k>  =  <R q, R k>                     R is orthonormal
        =  <rq, norm * centroid[code]>
```

so the dot product runs directly on the codes. The query is rotated **once**
per step; no cached token is ever touched by a rotation and no dense key or
value is ever materialised. The same holds on the value side, which is why the
output stays in the rotated domain and `R⁻¹` is folded into `W_o` offline.

## The pipeline

```
x ─┬─ W_q ─┐
   ├─ W_k ─┼─ 3 systolic arrays, fp16 × fp16 → fp32
   └─ W_v ─┘
          │
     fp32 → Q(qk_frac)        ← ops/convert.py, the ONE cast in the package
          │
        RoPE                  ← position enters here, before the cache
          │
     R (randomized Hadamard)  ← q, k and v, two rounds
          │
   quantize K,V → cache       ← never rotated or dequantized again
          │
     attend on codes          ← online or two-pass softmax, fixed point
          │
        W_o'                  ← R⁻¹ already folded in
```

## Architecture

**Ops emit values and a trace; the hardware model replays the trace to produce
cycles.** Timing is an observer, never a participant. No op can see a
`HardwareConfig`, so the hardware model cannot change a numeric result —
asserted directly in `tests/test_kernel.py::check_hardware_config_cannot_change_a_value`.

The consequence that matters: with timing off this is a plain attention module
you can graft into a model; with timing on it is an accelerator model. Same
code, not two implementations kept in agreement by hand.

| module | responsibility |
|---|---|
| `kernel.py` | composition — the only file that knows the pipeline order |
| `weights.py` | the four projections, with the offline folds applied |
| `ops/` | pure functions: project, rope, rotate, quantize, attention, output |
| `hw/` | cycles and bytes, replayed from a trace |
| `cache/` | `KVCache` interface; dense fp16 and compressed implementations |
| `numerics/` | `fp.py` (fp16×fp16→fp32) and `fixed.py` (**the only narrowing**) |
| `trace.py` | the seam between the two halves |
| `config.py` | one validated dataclass tree |

Three rules the package holds itself to:

1. **One float→fixed cast**, in `ops/convert.py`. A cast added wherever
   convenient is how two implementations of the same arithmetic come to
   disagree on inputs nobody tested.
2. **One narrowing module.** Intermediates are int64 and grow naturally; a
   value narrows only at a named `Q.rshift` / `Q.clamp` / `Q.from_float`.
3. **Prefill and decode are the same code.** `forward` takes `(tokens, hidden)`;
   one row is a decode step. There is no separate steady-state path.

## No calibration files

Rotating by a randomized Hadamard makes the channels near-Gaussian — that is
what the rotation is *for* — so the optimal scalar codebook afterwards is the
Lloyd–Max quantizer for `N(0,1)`, a pure function of the bit-width.
`Codebook.gaussian(bits)` derives it at construction.

That deletes a whole class of infrastructure: no calibration pass, no exported
ROM, no JSON that hardware and golden model must be kept in sync on, and no way
for a cache written under one calibration to be read under another. It is
checked against the published Max & Lloyd optima
(`tests/test_codebook.py::check_matches_published_lloyd_max`).

**The rotation uses one round, and that is a measured choice.** It was two for a
while, on the grounds that a single `HD` overshoots: applied to a one-hot vector
it gives `±1/√d` in every channel — a two-point distribution, excess kurtosis
exactly −2. Measured:

| input | 1 round | 2 rounds |
|---|---|---|
| one-hot (pathological) | −2.00 | −0.25 |
| 4 outlier channels ×30 | −0.87 | −0.14 |
| student-t, df = 2.5 | −0.26 | −0.08 |

All true, and **it does not predict quality**. Measured end to end on WikiText-2
(12 windows, paired, TinyLlama at 1024 context, 4b keys / 2b values), one round
beats two by **1.3% perplexity, significant at 2.8 standard errors**; three
rounds is indistinguishable from two.

The mechanism is probably the truncation: the `1/√d` scale is applied *inside*
each round, so a second round floors the value a second time, and that costs
more than the distributional gain returns. Kurtosis measures how Gaussian the
channels look, not what the quantizer can cash in — a proxy that resembles the
objective is not the objective, and the only way to find that out was to measure
the thing being optimised.

One round is also half the butterfly. See `experiments/ablate.py`.

## Measured on the real model

TinyLlama-1.1B, **all 22 layers grafted**, transformers 5.16.1, perplexity over
400 tokens of real text (`experiments/full_model.py`).

| configuration | ppl | vs base | B/tok/layer | vs fp16 |
|---|---|---|---|---|
| baseline (real fp32 attention) | 11.363 | — | 1024 | 1.00× |
| **kernel, dense (control)** | **11.363** | **1.000×** | 1024 | 1.00× |
| key 6b / value 6b | 11.510 | 1.013× | 400 | 2.56× |
| key 4b / value 4b | 11.589 | 1.020× | 272 | 3.76× |
| **key 4b / value 2b** | **12.112** | **1.066×** | **208** | **4.92×** |
| key 3b / value 2b | 13.902 | 1.223× | 176 | 5.82× |
| key 2b / value 2b | 34.725 | 3.056× | 144 | 7.11× |

**The control row is the important one.** With quantisation off, the kernel
reproduces the real model's perplexity exactly — so projection, the fp→fixed
cast, RoPE, the two-round rotation, the codes path and the `W_o` fold are all
right, and every row below it is attributable to compression alone. At the
single-layer level the same control matches `LlamaAttention` to **4.4e-4
relative, cosine 1.000000** (`experiments/single_layer.py`).

### Keys are expensive; values are nearly free

Sweeping one width with the other fixed at 4 bits (`experiments/ablate_bits.py`):

| bits | keys swept | values swept |
|---|---|---|
| 6 | 1.003× | 1.039× |
| 5 | 1.016× | 1.027× |
| 4 | 1.020× | 1.020× |
| 3 | **1.141×** | 1.044× |
| 2 | **2.351×** | **1.066×** |

Keys fall off a cliff below 4 bits — 3 bits costs 14%, 2 bits destroys the
model. Values degrade gracefully: going all the way to 2 bits costs 6.6%.

So the two planes should **not** be the same width, and the value plane is
where the bits should come from first. `key_bits=4, value_bits=2` is the best
point measured: **4.92× at +6.6% perplexity**. A symmetric format at the same
size (3b/2b, 5.82×) costs +22.3% — over three times the damage for 18% more
compression.

### It generates text

`experiments/generate.py`, greedy decoding, all 22 layers grafted. At 4b/3b the
output is nearly token-identical to the baseline; at 3b/2b it diverges but
stays coherent and on-topic.

## What the hardware model says

TinyLlama shape (2048 hidden, 32 heads, 4 KV heads, d=64), 3-bit keys / 2-bit
values, 128×128 arrays, 256-bit port, one decode step at context 513:

```
traffic    18,874,368 B weights + 722,480 B KV = 19,596,848 B (96% weights)
q_array    327,424 cyc  bound by weights  util 0.08%
```

Two findings fall straight out, and both are asserted as tests rather than
described:

- **Decode projection is a GEMV, so every array is weight-bound at ~0.08%
  utilisation.** Every weight is used exactly once, so arithmetic intensity is
  1 MAC per weight. Growing the array 16× buys under 2×. Widening the *port*
  is what moves the bound. (`test_hardware.py::check_gemv_is_weight_bound_and_no_array_size_fixes_it`)
- **Weight streaming is 96% of decode traffic at context 513**, 26× the KV
  cache. Compressing the cache 5.8× removes 15% of total traffic — real, but
  not the headline the byte ratio alone suggests.

The crossover where an array stops being weight-bound is computable, not a
matter of taste: a 128×128 tile on a 256-bit port needs a batch of **1024** to
saturate, which is far more than decode ever has.

## Running it

```bash
.venv/bin/python -m kernel.tests.run          # 74 checks, no pytest needed
.venv/bin/python -m kernel.demo 512           # cost of a decode step at context 512
```

Against the real model (needs `transformers`; downloads TinyLlama on first run):

```bash
.venv/bin/python kernel/experiments/probe_contract.py   # what a decoder layer passes
.venv/bin/python kernel/experiments/single_layer.py     # one layer vs real LlamaAttention
.venv/bin/python kernel/experiments/full_model.py 400   # all 22 layers, perplexity
.venv/bin/python -m kernel.experiments.ablate_bits 400  # keys vs values
.venv/bin/python kernel/experiments/generate.py         # actual text
```

## Grafting into a real model

```python
from kernel.adapters.torch_llama import graft, compare_against_original

original = graft(model, layers=[10], quant=QuantConfig(key_bits=3, value_bits=2))
```

torch is imported lazily and is not a package dependency.

## Known limits — carry these, do not drop them

1. **The adapter is verified against transformers 5.16.1 and nothing else.**
   That contract moves: 5.x nests `rope_theta` inside `rope_parameters`, and the
   layer passes `position_ids` rather than `cache_position`. Re-run
   `experiments/single_layer.py` after any version bump — its dense-mode row is
   the plumbing check, and it must come back at cosine 1.000000.
2. **`attention_mask` is ignored** and batch > 1 is refused rather than
   tolerated. The kernel is causal by construction, which covers decoding, but
   not padded batches or custom masks.
3. **Prefill and decode are not bit-identical** — they agree to ~2e-7 relative.
   The fp32 GEMM blocks differently for different batch sizes, so its
   accumulation order changes. Everything downstream of the fixed-point
   boundary is exact; this residual is fp32 reassociation and nothing else.
4. **Online and two-pass softmax are not bit-identical either**, and must not
   be asserted to be. The online form truncates the accumulator once per
   rescale event. `forward` uses two-pass; the online form is what hardware
   would run, and the gap is measured, not assumed.
5. **`Weights.random` is stimulus, not a quality claim.** Nothing about
   perplexity can be argued from a run on synthetic weights — `demo.py` uses
   them, the experiments use the real checkpoint.
6. **Perplexity is over 400 tokens of one passage.** Enough to separate a 2%
   effect from a 22% one, which is what the tables above are used for. NOT
   enough for a publishable quality claim, and the passage is in-domain for the
   text the sweep discusses. A real evaluation needs a held-out corpus.
7. **The grafted model writes zeros into `past_key_values`.** That object is
   bookkeeping only — the kernel never reads it — but it means the model's own
   cache holds nothing useful, and anything that inspects it will be misled.
8. **The cycle model is post-synthesis-free.** It is an analytical schedule
   model with a stated dataflow (output-stationary, tiles not overlapped), not
   a number from a synthesis tool. Its value is in showing which bound binds.
9. **`d_head` must be a power of two.** The rotation is a Walsh–Hadamard
   butterfly and has no non-power-of-two form. This is an architectural limit,
   not an unimplemented case.
