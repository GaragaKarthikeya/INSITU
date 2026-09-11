"""Pareto plot: perplexity vs. KV cache bytes/token/layer, Llama 3.2 1B.

The 25-config (key_bits, value_bits) grid, WikiText-2, 2,048 tokens, all 16
layers, paths A ("rebuild the dense key, then dot") and B ("no rebuild, table
form" -- what the hardware implements). Source: `plan.MD`, the table under
"All twenty-five are paired" (identical numbers to
`experiments/results/paths/sweep_AB.log`).

`row` in that table is bytes/token for one KV head; Llama 3.2 1B has 8 KV
heads, so `bytes_per_token_per_layer = row_bytes * 8` -- this reproduces the
measured "416 B/token/layer" figure plan.MD reports separately for k4v2
(52 B x 8 = 416 B).

    python -m kernel.experiments.plot_pareto_llama32 \\
        --out experiments/results/pareto_llama32_ppl_vs_bytes.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

NUM_KV_HEADS = 8  # Llama 3.2 1B

# (key_bits, value_bits, row_bytes, ppl_A, ppl_B) -- from plan.MD
GRID = [
    (2, 2, 36, 87.9170, 94.5112),
    (2, 3, 44, 82.5741, 74.2553),
    (3, 2, 44, 11.5994, 11.3830),
    (2, 4, 52, 66.1892, 69.8478),
    (3, 3, 52, 10.0402, 10.1547),
    (4, 2, 52, 8.5463, 8.7209),
    (2, 5, 60, 64.6602, 66.0630),
    (3, 4, 60, 9.6539, 9.8474),
    (4, 3, 60, 7.6965, 7.7710),
    (5, 2, 60, 7.9426, 8.0035),
    (2, 6, 68, 77.2041, 71.3001),
    (3, 5, 68, 9.5958, 9.5943),
    (4, 4, 68, 7.6146, 7.5510),
    (5, 3, 68, 7.3741, 7.3614),
    (6, 2, 68, 7.9608, 7.9541),
    (3, 6, 76, 9.6164, 9.5107),
    (4, 5, 76, 7.5806, 7.5603),
    (5, 4, 76, 7.2678, 7.2794),
    (6, 3, 76, 7.2963, 7.3116),
    (4, 6, 84, 7.5225, 7.6375),
    (5, 5, 84, 7.2637, 7.2069),
    (6, 4, 84, 7.2257, 7.2192),
    (5, 6, 92, 7.2250, 7.2572),
    (6, 5, 92, 7.1531, 7.1840),
    (6, 6, 100, 7.1506, 7.1668),
]

FP16_BASELINE = 7.1439


def pareto_frontier(points: list[dict], y_key: str) -> list[dict]:
    frontier = []
    for p in points:
        dominated = any(
            q is not p and q["bytes"] <= p["bytes"] and q[y_key] <= p[y_key]
            and (q["bytes"] < p["bytes"] or q[y_key] < p[y_key])
            for q in points
        )
        if not dominated:
            frontier.append(p)
    return sorted(frontier, key=lambda p: p["bytes"])


def best_per_bytes(points: list[dict]) -> list[dict]:
    """Collapse configs sharing the same bytes/token to their lowest-ppl one.

    (key_bits, value_bits) pairs with the same total, e.g. (4,2) and (2,4),
    land at the same row size -- they're the same storage cost, so only the
    better-perplexity split is worth showing.
    """
    best: dict[float, dict] = {}
    for p in points:
        cur = best.get(p["bytes"])
        if cur is None or p["B"] < cur["B"]:
            best[p["bytes"]] = p
    return sorted(best.values(), key=lambda p: p["bytes"])


def plot(out: Path) -> None:
    points = [
        {"label": f"k{kb}v{vb}", "bytes": row * NUM_KV_HEADS, "A": a, "B": b}
        for kb, vb, row, a, b in GRID
    ]
    points = best_per_bytes(points)

    frontier_b = pareto_frontier(points, "B")
    frontier_b_labels = {p["label"] for p in frontier_b}

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter([p["bytes"] for p in points], [p["B"] for p in points],
               marker="o", color="tab:blue", s=35, label="path B (table form, hw)", zorder=3)

    for p in points:
        weight = "bold" if p["label"] in frontier_b_labels else "normal"
        ax.annotate(p["label"], (p["bytes"], p["B"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8,
                    fontweight=weight)

    ax.plot([p["bytes"] for p in frontier_b], [p["B"] for p in frontier_b],
            color="black", linewidth=1, linestyle="--", zorder=1,
            label="Pareto frontier (path B)")

    ax.set_xlabel("KV cache bytes / token / layer (8 KV heads)")
    ax.set_ylabel("Perplexity (WikiText-2, 2,048 tok, 16 layers)")
    ax.set_title("Llama 3.2 1B: perplexity vs. KV cache footprint, 25 key/value widths")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nPareto-optimal configs (path B):")
    for p in frontier_b:
        print(f"  {p['label']:6s} {p['bytes']:6.0f} B/tok/layer  ppl(B)={p['B']:.4f}"
              f"  ({p['B']/FP16_BASELINE:.3f}x fp16)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                     default=Path("experiments/results/pareto_llama32_ppl_vs_bytes.png"))
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
