"""Pareto plot: KV-cache bytes/token/layer vs. perplexity, across quant configs.

Reads one or more `ablate.py`-style `results.json` files (each a dict of
variant name -> {"ppl": float, "bytes_per_token_per_layer": int|None, ...})
and plots every (bytes_per_token_per_layer, ppl) pair on one axis, with the
non-dominated (Pareto-optimal) configs highlighted and connected.

Bytes/token is used rather than device cycles/latency because the cache scan
engine retires one row per cycle regardless of row size below the DRAM/
interface bound (`CacheConfig.scan_cycles`, `config.py`) -- two configs with
different storage can tie in latency in that regime, which would collapse
real tradeoffs on a latency axis. Bytes/token has no such floor.

    python -m kernel.experiments.plot_pareto \\
        experiments/results/ablate/grid-c1024-tinyllama-1.1b-chat-v1.0/results.json \\
        experiments/results/qwen3-8b/results.json \\
        --out experiments/results/pareto_ppl_vs_bytes.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_points(paths: list[Path]) -> list[dict]:
    """One dict per variant with a usable bytes/token figure, tagged by source."""
    points = []
    for p in paths:
        data = json.loads(p.read_text())
        source = p.parent.name
        for variant, rec in data.items():
            bpt = rec.get("bytes_per_token_per_layer")
            ppl = rec.get("ppl")
            if bpt is None or ppl is None:
                continue  # e.g. "baseline"/"meta": no cache format to size
            points.append({"source": source, "variant": variant,
                            "bytes": bpt, "ppl": ppl})
    return points


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Non-dominated points: no other point has both lower-or-equal bytes and
    lower-or-equal ppl, with at least one strictly lower."""
    frontier = []
    for p in points:
        dominated = any(
            q is not p
            and q["bytes"] <= p["bytes"] and q["ppl"] <= p["ppl"]
            and (q["bytes"] < p["bytes"] or q["ppl"] < p["ppl"])
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p["bytes"])


def plot(points: list[dict], out: Path) -> None:
    frontier = pareto_frontier(points)
    frontier_keys = {(p["source"], p["variant"]) for p in frontier}

    sources = sorted({p["source"] for p in points})
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % 10) for i, s in enumerate(sources)}

    fig, ax = plt.subplots(figsize=(7, 5))

    for s in sources:
        pts = [p for p in points if p["source"] == s]
        ax.scatter([p["bytes"] for p in pts], [p["ppl"] for p in pts],
                   color=color[s], label=s, s=40, zorder=3)

    for p in points:
        marker = "o" if (p["source"], p["variant"]) in frontier_keys else "x"
        ax.annotate(p["variant"], (p["bytes"], p["ppl"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)
        if marker == "x":
            ax.scatter([p["bytes"]], [p["ppl"]], marker="x",
                       color=color[p["source"]], s=30, zorder=2)

    ax.plot([p["bytes"] for p in frontier], [p["ppl"] for p in frontier],
            color="black", linewidth=1, linestyle="--", zorder=1,
            label="Pareto frontier")

    ax.set_xlabel("KV cache bytes / token / layer")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs. KV cache footprint")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nPareto-optimal configs:")
    for p in frontier:
        print(f"  {p['source']:35s} {p['variant']:8s} "
              f"{p['bytes']:6.0f} B/tok/layer  ppl={p['ppl']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path,
                     help="one or more results.json files")
    ap.add_argument("--out", type=Path,
                     default=Path("experiments/results/pareto_ppl_vs_bytes.png"))
    args = ap.parse_args()

    points = load_points(args.results)
    if not points:
        raise SystemExit("no (bytes_per_token_per_layer, ppl) pairs found")
    plot(points, args.out)


if __name__ == "__main__":
    main()
