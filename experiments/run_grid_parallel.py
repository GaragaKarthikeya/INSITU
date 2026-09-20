"""Run a config grid across several OS processes instead of one.

    python -m kernel.experiments.run_grid_parallel \\
        --model kernel/models/Qwen2.5-1.5B --context 2048 --windows 12 \\
        --configs k2v2-A,k2v2-B,...  --workers 6 --dir results/ab-grid-qwen

Why processes and not threads
------------------------------
`wikitext_ppl.py` spends most of a window's wall clock in `ops/attention.py`
doing fixed-point bookkeeping and an exp-table gather -- plain NumPy ufuncs,
which run on one core each and do not parallelise themselves just because
more cores exist. Threading that code correctly means auditing
`CompressedAttention`'s mutable per-call state (`_q_unrot`, `baseline_calls`)
for races and re-proving every bit-exactness pin under concurrent calls --
real work, on a hot path several tests already pin tightly.

Splitting the CONFIG LIST across processes instead needs none of that. Each
worker is an ordinary, unmodified `wikitext_ppl.py` run over a disjoint slice
of the grid, in its own directory, so there is nothing to make thread-safe --
the OS scheduler is what spreads the work across cores. The trade is
per-worker fixed costs (model load, grafting) paid once per worker rather
than once per run, which is worth it once a worker's slice is more than a
few configs.

Why every worker gets `baseline` too
-------------------------------------
`wikitext_ppl.py` reports each config against `state["baseline"]`, and the
dense-control check (`log.check_dense_control`) needs it in the same
process's `state` if `dense` is one of the configs being run. Computing it
once per worker is one extra window's worth of a cheap config, and it is the
only way to keep every worker's `results.json` self-contained and directly
runnable on its own. The merge step below cross-checks that every worker's
baseline agrees, rather than assuming it.

BLAS thread count is capped per worker
---------------------------------------
The big matmuls in `attend_causal_batch` are already multi-threaded (this
project's NumPy links OpenBLAS). Leaving `OPENBLAS_NUM_THREADS` at its
default -- every core -- inside every worker would have `--workers N`
processes each independently trying to claim all 24 cores for their matmuls,
which oversubscribes and can run slower than one worker. Each worker's BLAS
thread count is capped to `cores // workers` so the two axes of parallelism
(processes across cores, BLAS across a matmul) divide the machine instead of
fighting over it.

GPU memory is not divided automatically
-----------------------------------------
Each worker loads its own copy of the model. On `--device cuda` that is one
full set of weights per worker in VRAM -- this project's 16 GB card holds
roughly two fp32 copies of a 1-2B model before the allocator starts failing,
so `--workers` above 2 on `--device cuda` will very likely OOM. `--device
cpu` has no such ceiling beyond system RAM, at the cost of a slower forward
pass; given the forward pass is a small fraction of a window's cost next to
the attention bookkeeping above, CPU-only workers are usually the better
trade once `--workers` is large.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
KERNEL_ROOT = HERE.parents[2]          # .../ containing the `kernel` package


def split_round_robin(items: list[str], n: int) -> list[list[str]]:
    """`n` roughly-equal slices, round-robin so a slow config doesn't stack."""
    buckets: list[list[str]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def merge(base_dir: Path, n_workers: int) -> dict:
    """Combine every worker's `results.json` into one, checked rather than trusted.

    Not a blind union: two workers reporting a different `baseline` (or a
    different `meta`) means they ran against different models, contexts or
    corpora, and averaging that together would silently produce a number that
    corresponds to nothing. That is refused rather than merged.
    """
    merged: dict = {}
    meta_ref = None
    baseline_ref = None
    for i in range(n_workers):
        p = base_dir / f"_worker{i}" / "results.json"
        if not p.exists():
            continue
        state = json.loads(p.read_text())
        meta = state.pop("meta", None)
        if meta is not None:
            if meta_ref is None:
                meta_ref = meta
            elif meta != meta_ref:
                raise ValueError(
                    f"worker {i}'s meta disagrees with worker 0's: {meta} != "
                    f"{meta_ref}. Workers must share --model/--context/--windows."
                )
        base = state.get("baseline", {}).get("ppl")
        if base is not None:
            if baseline_ref is None:
                baseline_ref = base
            elif abs(base - baseline_ref) > 1e-6:
                raise ValueError(
                    f"worker {i}'s baseline ppl ({base}) disagrees with an "
                    f"earlier worker's ({baseline_ref}). Same model and windows "
                    f"should give the identical fp32 baseline; something differs."
                )
        for name, rec in state.items():
            if name == "baseline":
                continue           # every worker computes it; checked above instead
            if name in merged and merged[name] != rec:
                raise ValueError(
                    f"config {name!r} was reported by more than one worker "
                    f"with different results; the config list given to each "
                    f"worker must be disjoint."
                )
            merged[name] = rec
    if baseline_ref is not None:
        # Any worker's copy will do -- the check above already confirmed they
        # agree to 1e-6 in ppl, and `seconds` (the only field that legitimately
        # differs, being wall clock) is not something a caller reads off this.
        for i in range(n_workers):
            p = base_dir / f"_worker{i}" / "results.json"
            if p.exists():
                merged["baseline"] = json.loads(p.read_text())["baseline"]
                break
    if meta_ref is not None:
        merged = {"meta": meta_ref, **merged}
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None,
                    help="required unless --merge-only")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--configs", default=None,
                    help="comma-separated config names, as wikitext_ppl.py "
                         "accepts them; split across workers, round-robin. "
                         "'baseline' is added to every worker automatically "
                         "and need not be listed. Required unless --merge-only.")
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--dir", required=True,
                    help="base output directory; each worker writes to "
                         "<dir>/_worker<i>, and --merge-only reads them back")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--pilot-chunk", type=int, default=2)
    ap.add_argument("--openblas-threads", type=int, default=None,
                    help="override the per-worker BLAS thread cap; default "
                         "is cpu_count() // workers, at least 1")
    ap.add_argument("--merge-only", action="store_true",
                    help="don't launch anything; merge whatever the workers "
                         "in --dir have written so far (safe to run while "
                         "workers are still going, for a partial table)")
    a = ap.parse_args(argv)

    base_dir = Path(a.dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not a.merge_only:
        if not a.configs:
            raise SystemExit("--configs is required unless --merge-only")
        if not a.model:
            raise SystemExit("--model is required unless --merge-only")
        wanted = [c.strip() for c in a.configs.split(",") if c.strip()]
        wanted = [c for c in wanted if c != "baseline"]
        if not wanted:
            raise SystemExit("--configs has nothing besides baseline to split")
        chunks = split_round_robin(wanted, a.workers)
        empty = [i for i, c in enumerate(chunks) if not c]
        if empty:
            raise SystemExit(
                f"--workers {a.workers} exceeds the {len(wanted)} configs given; "
                f"worker(s) {empty} would have nothing to do"
            )

        threads = a.openblas_threads or max(1, multiprocessing.cpu_count() // a.workers)
        if a.device == "cuda" and a.workers > 2:
            print(f"warning: {a.workers} workers on --device cuda will each load "
                  f"a full copy of {a.model} into VRAM; this is very likely to "
                  f"OOM on a single GPU above 2 workers. Consider --device cpu.",
                  file=sys.stderr)

        procs = []
        for i, chunk in enumerate(chunks):
            worker_dir = base_dir / f"_worker{i}"
            cmd = [
                sys.executable, "-m", "kernel.experiments.wikitext_ppl",
                "--model", a.model, "--context", str(a.context),
                "--configs", "baseline," + ",".join(chunk),
                "--dir", str(worker_dir),
                "--device", a.device, "--dtype", a.dtype,
                "--chunk", str(a.chunk), "--pilot-chunk", str(a.pilot_chunk),
            ]
            if a.windows is not None:
                cmd += ["--windows", str(a.windows)]
            env = dict(os.environ)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
            env["OMP_NUM_THREADS"] = str(threads)
            worker_dir.mkdir(parents=True, exist_ok=True)
            log_path = worker_dir / "launcher_stdout.log"
            print(f"worker {i}: {len(chunk)} configs {chunk}, "
                  f"{threads} BLAS threads -> {log_path}")
            f = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(cmd, cwd=str(KERNEL_ROOT), env=env,
                                    stdout=f, stderr=subprocess.STDOUT)
            procs.append((proc, f))

        t0 = time.time()
        exit_codes = []
        for i, (proc, f) in enumerate(procs):
            code = proc.wait()
            f.close()
            exit_codes.append(code)
            print(f"worker {i} exited {code} after {time.time() - t0:.0f}s total")
        bad = [i for i, c in enumerate(exit_codes) if c not in (0,)]
        if bad:
            print(f"worker(s) {bad} did not exit 0 -- see their "
                  f"launcher_stdout.log before trusting the merge", file=sys.stderr)

    merged = merge(base_dir, a.workers)
    (base_dir / "results.json").write_text(json.dumps(merged, indent=1))
    n_configs = len([k for k in merged if k != "meta"])
    print(f"\nmerged {n_configs} configs from {a.workers} workers into "
          f"{base_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
