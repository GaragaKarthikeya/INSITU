"""WikiText-2 perplexity for the compressed-KV kernel, at a publishable protocol.

Protocol
--------
Stated here so a number from this can be compared to a published table without
guessing:

  * `wikitext-2-raw-v1`, test split, joined with blank lines as it ships.
  * Tokenised once as a single stream, then cut into non-overlapping windows of
    `--context` tokens. A trailing partial window is dropped, not padded.
  * Each window is an independent sequence: the KV cache is reset between them.
  * Perplexity is `exp(sum NLL / sum scored tokens)` over all windows.

Interleaved, not one config at a time
-------------------------------------
Configurations take turns in chunks rather than running end to end. Running
them sequentially means that at hour 30 you have two finished columns and three
empty ones; interleaving means that at any moment you have a complete but
shallow table -- every configuration at the same window count. That is both
more informative while it runs and directly reportable if you stop early.

The cost is re-grafting once per chunk (the `W_o` fold is a 4096-cube float64
matmul per layer, about 20 s for a 36-layer model), which at a chunk of 4 is
under 1% of the run.

The first round is a pilot
--------------------------
The first chunk is deliberately small, so every configuration is exercised
within the first couple of hours. A configuration that crashes, or a graft that
does not reproduce the baseline, then surfaces at hour 2 instead of hour 40.

The saturation gate rides along
-------------------------------
The 24-bit wire lane between the host and the PL is Q8.16, and the evidence for
it is a dynamic-range peak of 21.06 over 400 tokens of one passage. That is not
enough text to lock a wire format on, and `FixedFormat.saturate=True` clamps
rather than wraps -- so a saturating element is silent quality loss, not a
crash. Nothing would fail; the numbers would just quietly get worse.

So every window this runs also aggregates `StepReport.conversion` across all
layers, and the table carries `sat` (the fraction of Q/K/V elements that hit the
rail) and `max|x|` (the largest float that reached the cast). `sat` must be 0.

This costs nothing: `ConversionStats` is already computed inside every
`forward`, and was simply being discarded at the adapter boundary.

What this does not change
-------------------------
Nothing here touches the hardware model. Byte counts, cycle accounting and the
trace are exactly what a single-token run produces; this file only decides what
text goes in and in what order. The saturation figures are read off reports the
kernel already produced -- an observation, not a code path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel import ModelConfig, QuantConfig
from kernel.ops.convert import ConversionStats
from kernel.adapters.torch_llama import graft, ungraft
from kernel.experiments.runlog import (Alarm, RunLog, STOPPED_EXIT,
                                       read_json_resilient, write_json_atomic)

# Set by a SIGTERM or SIGINT handler. Checked at every window boundary, so a
# shutdown or a Ctrl-C stops between windows with a valid checkpoint rather
# than in the middle of one. Because the checkpoint is written atomically, even
# a kill -9 costs at most the window in flight.
STOPPING = False


def _install_signal_handlers(log: RunLog) -> None:
    import signal

    def handler(signum, frame):
        global STOPPING
        name = signal.Signals(signum).name
        if STOPPING:                       # a second signal means "now"
            log.event("stop_forced", signal=name)
            raise SystemExit(1)
        STOPPING = True
        print(f"\n{name} received -- finishing the current window, then stopping. "
              f"Send it again to stop immediately.", flush=True)
        log.event("stop_requested", signal=name)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass

# name -> (mode, QuantConfig, max_windows or None for "all")
CONFIGS = {
    "baseline": (None, None, None),
    "dense":    ("dense", QuantConfig(), 12),
    "k4v4":     ("compressed", QuantConfig(key_bits=4, value_bits=4), None),
    "k4v3":     ("compressed", QuantConfig(key_bits=4, value_bits=3), None),
    "k4v2":     ("compressed", QuantConfig(key_bits=4, value_bits=2), None),
    "k3v2":     ("compressed", QuantConfig(key_bits=3, value_bits=2), None),
}


def load_windows(tok, context, limit):
    from datasets import load_dataset
    # The namespaced id. `datasets` 5.x parses dataset ids as HF URIs and the
    # legacy bare "wikitext" fails that parse; "Salesforce/wikitext" is the same
    # data at its canonical location.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = ids.shape[0] // context
    return ids, (min(n, limit) if limit else n), ids.shape[0]


def attach_heartbeat(model, log: RunLog):
    """One tick per transformer layer, so liveness is visible within seconds.

    A per-window log line is 14 minutes apart; a wedged process and a slow one
    look identical at that resolution.
    """
    n = len(model.model.layers)
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(
            lambda mod, inp, out, i=i: log.heartbeat(layer=i + 1, layers_total=n)))
    return handles


def conversion_stats(model) -> ConversionStats:
    """Sum `StepReport.conversion` over every grafted layer.

    Returns an empty `ConversionStats` when nothing is grafted -- the baseline
    config runs stock attention and never reaches the fp -> fixed cast, so
    there is genuinely nothing to report rather than a zero to interpret.
    """
    total = ConversionStats()
    for layer in model.model.layers:
        r = getattr(layer.self_attn, "last_report", None)
        if r is not None:
            total.merge(r.conversion)
    return total


def reset_all(model):
    for layer in model.model.layers:
        a = layer.self_attn
        if hasattr(a, "reset"):
            a.reset()


def table(state, order):
    base = state.get("baseline", {}).get("ppl")
    out = [f"{'config':<10} {'ppl':>9} {'vs base':>9} {'windows':>8} "
           f"{'B/tok/layer':>12} {'sat':>9} {'max|x|':>8} {'hours':>7}"]
    for name in order:
        r = state.get(name)
        if not r or "ppl" not in r:
            continue
        ratio = f"{r['ppl'] / base:8.4f}x" if base else "       --"
        sat = r.get("saturation_rate")
        mx = r.get("max_abs_float")
        out.append(f"{name:<10} {r['ppl']:9.4f} {ratio} {r['windows']:8d} "
                   f"{str(r.get('bytes_per_token_per_layer', '-')):>12} "
                   f"{'--' if sat is None else f'{sat:9.2e}'} "
                   f"{'--' if mx is None else f'{mx:8.3f}'} "
                   f"{r['seconds'] / 3600:7.2f}")
    return "\n".join(out)


@torch.no_grad()
def run_window(model, window):
    ids = window.unsqueeze(0)
    out = model(ids, labels=ids)
    n = ids.shape[1] - 1                 # the first token has no prediction
    return float(out.loss) * n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--configs", default="baseline,dense,k4v4,k4v3,k4v2,k3v2")
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--dir", default="kernel/experiments/results/qwen3-8b")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--pilot-chunk", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--dense-tol", type=float, default=0.005)
    ap.add_argument("--rss-warn", type=float, default=44.0)
    ap.add_argument("--rss-abort", type=float, default=46.5)
    ap.add_argument("--need-gb", type=float, default=38.0,
                    help="refuse to start with less available memory than this")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    d = Path(args.dir)
    log = RunLog(d)
    _install_signal_handlers(log)
    results_path = d / "results.json"
    state, source = read_json_resilient(results_path)
    if source is not None:
        done = {k: v.get("windows", 0) for k, v in state.items() if k != "meta"}
        print(f"resuming from {source.name}: {done}", flush=True)
        log.event("resume", source=source.name, done=done)

    log.set(phase="loading", model=args.model, context=args.context, dtype=args.dtype)
    log.event("start", model=args.model, context=args.context, dtype=args.dtype,
              configs=args.configs, argv=sys.argv[1:])

    # Refuse to start a run that obviously cannot finish, rather than
    # discovering it eight hours in.
    log.preflight(need_gb=args.need_gb, need_disk_gb=1.0)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids, n_windows, n_tokens = load_windows(tok, args.context, args.windows)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).eval()
    m = ModelConfig.from_hf(model.config)
    layers = list(range(model.config.num_hidden_layers))
    attach_heartbeat(model, log)

    meta = dict(model=args.model, context=args.context, dtype=args.dtype,
                corpus="Salesforce/wikitext:wikitext-2-raw-v1/test", n_tokens=n_tokens,
                n_windows=n_windows, layers=len(layers), heads=m.num_heads,
                kv_heads=m.num_kv_heads, head_dim=m.head_dim,
                protocol="non-overlapping windows, cache reset per window")
    state.setdefault("meta", {}).update(meta)
    log.set(**meta, phase="pilot")
    log.event("model_loaded", **meta)
    print(f"corpus  {n_tokens:,} tokens -> {n_windows} windows of {args.context}")
    print(f"model   {args.model}  {len(layers)}L {m.num_heads}H "
          f"{m.num_kv_heads}KV d{m.head_dim} {args.dtype}\n", flush=True)

    order = args.configs.split(",")
    targets = {n: min(CONFIGS[n][2] or n_windows, n_windows) for n in order}
    for n in order:
        state.setdefault(n, {"nll": 0.0, "tokens": 0, "windows": 0, "seconds": 0.0})
        # Per-window log-perplexity. Needed for the dense-control check (which
        # must compare the same windows) and for the paired error bars every
        # reported number should carry. Storing only a running total made both
        # impossible.
        state[n].setdefault("per_window", [])

    chunk, round_i = args.pilot_chunk, 0
    try:
        while any(state[n]["windows"] < targets[n] for n in order):
            round_i += 1
            phase = "pilot" if round_i == 1 else "main"
            log.set(phase=phase, round=round_i, chunk=chunk)

            for name in order:
                rec, target = state[name], targets[name]
                if rec["windows"] >= target:
                    continue
                stop = min(rec["windows"] + chunk, target)
                mode, quant, _ = CONFIGS[name]

                original = None
                if mode is not None:
                    log.set(config=name, activity="grafting")
                    t0 = time.time()
                    original = graft(model, layers, quant=quant,
                                     capacity=args.context + 8,
                                     compressed=(mode == "compressed"))
                    rec["bytes_per_token_per_layer"] = \
                        model.model.layers[0].self_attn.kernel.cache.bytes_per_token
                    log.event("graft", config=name, seconds=round(time.time() - t0, 1),
                              bytes_per_token_per_layer=rec["bytes_per_token_per_layer"])
                try:
                    for w in range(rec["windows"], stop):
                        t0 = time.time()
                        log.set(config=name, window=w + 1, window_target=target,
                                activity="forward")
                        if original is not None:
                            reset_all(model)
                        nll, n = run_window(model, ids[w * args.context:
                                                       (w + 1) * args.context])
                        log.check_loss(name, w, nll)
                        rec["nll"] += nll
                        rec["tokens"] += n
                        rec["windows"] = w + 1
                        rec["per_window"].append(nll / n)
                        rec["seconds"] += time.time() - t0
                        rec["ppl"] = float(np.exp(rec["nll"] / rec["tokens"]))

                        # The saturation gate. Accumulated across windows, not
                        # reset, so the rate is over every element the cast has
                        # ever seen -- which is the population the wire format
                        # has to hold. A rate that is zero on window 1 and
                        # non-zero on window 40 is exactly the failure this is
                        # here to catch, so it must not be a per-window figure.
                        c = conversion_stats(model)
                        if c.n_values:
                            rec["n_values"] = rec.get("n_values", 0) + c.n_values
                            rec["n_saturated"] = rec.get("n_saturated", 0) + c.n_saturated
                            rec["max_abs_float"] = max(rec.get("max_abs_float", 0.0),
                                                       c.max_abs_float)
                            rec["saturation_rate"] = \
                                rec["n_saturated"] / rec["n_values"]
                            if rec["n_saturated"]:
                                log.event("saturation", config=name, window=w + 1,
                                          rate=rec["saturation_rate"],
                                          max_abs_float=rec["max_abs_float"],
                                          note="Q8.16 wire lane CLAMPED. Silent "
                                               "quality loss -- the 24-bit lane "
                                               "width is not safe as it stands.")

                        write_json_atomic(results_path, state)
                        done = sum(state[c]["windows"] for c in order)
                        total = sum(targets[c] for c in order)
                        eta = (time.time() - log.started) / max(done, 1) * (total - done)
                        log.set(ppl={c: round(state[c].get("ppl", 0), 4) for c in order},
                                done_windows=done, total_windows=total,
                                eta_h=round(eta / 3600, 2))
                        log.event("window", config=name, window=w + 1, target=target,
                                  ppl=round(rec["ppl"], 5),
                                  seconds=round(time.time() - t0, 1),
                                  eta_h=round(eta / 3600, 2))
                        sat = ("" if "saturation_rate" not in rec else
                               f"  sat {rec['saturation_rate']:.2e} "
                               f"max|x| {rec['max_abs_float']:6.3f}")
                        print(f"{name:<9} w{w + 1:3d}/{target}  ppl {rec['ppl']:8.4f}  "
                              f"{time.time() - t0:6.1f}s  ETA {eta / 3600:5.2f} h{sat}",
                              flush=True)
                        log.check_memory(args.rss_abort, args.rss_warn)
                        if STOPPING:
                            log.set(phase="stopped")
                            log.event("stopped", config=name, window=w + 1,
                                      note="clean stop at a window boundary; "
                                           "rerun the same command to resume")
                            print("\nstopped cleanly. Rerun the same command "
                                  "to resume from here.", flush=True)
                            raise SystemExit(STOPPED_EXIT)
                finally:
                    if original is not None:
                        ungraft(model, original)

                # The check that decides whether anything else is worth running.
                if name == "dense" and "baseline" in state:
                    bw = state["baseline"].get("per_window", [])
                    dw = state["dense"].get("per_window", [])
                    if bw and dw:
                        log.check_dense_control(bw, dw, args.dense_tol)
                        shared = min(len(bw), len(dw))
                        log.event("dense_control_ok", shared_windows=shared,
                                  baseline=round(float(np.exp(np.mean(bw[:shared]))), 5),
                                  dense=round(float(np.exp(np.mean(dw[:shared]))), 5))

            if round_i == 1:
                chunk = args.chunk
                log.event("pilot_complete", table=table(state, order))
                print("\n--- PILOT COMPLETE ---\n" + table(state, order) + "\n",
                      flush=True)

        log.set(phase="done")
        log.event("done", table=table(state, order))
    except Alarm:
        raise
    except Exception as e:
        log.event("crash", error=f"{type(e).__name__}: {e}")
        raise

    print("\n" + table(state, order))


if __name__ == "__main__":
    main()
