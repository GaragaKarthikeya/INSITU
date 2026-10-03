"""Pareto plot: perplexity vs. KV cache bytes/token/layer, Llama 3.2 1B.

The 25-config (key_bits, value_bits) grid, WikiText-2, all 16 layers, paths A
("rebuild the dense key, then dot") and B ("no rebuild, table form" -- what the
hardware implements).

Every perplexity here is the mean over **12 windows**, which supersedes the
earlier single-run grid (baseline 7.1439) that `plan.MD` records. None of the
conclusions moved: the best config at each plane count is unchanged, and keys
still beat values at every bit budget. The numbers themselves all did.

The 12 windows are 24,564 of WikiText-2 test's 289,077 tokens, so the baseline
is a baseline for this grid and is **not** comparable to published
full-test-set numbers for this model.

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

# (key_bits, value_bits, row_bytes, ppl_A, ppl_B)
# Perplexity is the mean over 12 WikiText-2 windows, not a single run.
GRID = [
    (2, 2, 36, 102.1468, 103.3080),
    (2, 3, 44, 91.4258, 91.7121),
    (3, 2, 44, 14.6583, 14.6055),
    (2, 4, 52, 84.2349, 85.3821),
    (3, 3, 52, 13.0430, 13.0365),
    (4, 2, 52, 11.0146, 11.1043),
    (2, 5, 60, 83.3800, 82.7317),
    (3, 4, 60, 12.6855, 12.6456),
    (4, 3, 60, 10.1944, 10.1830),
    (5, 2, 60, 10.5020, 10.5001),
    (2, 6, 68, 82.4657, 83.5500),
    (3, 5, 68, 12.4893, 12.5388),
    (4, 4, 68, 10.0136, 10.0084),
    (5, 3, 68, 9.7404, 9.7362),
    (6, 2, 68, 10.3528, 10.3308),
    (3, 6, 76, 12.4504, 12.6234),
    (4, 5, 76, 9.9556, 9.9409),
    (5, 4, 76, 9.5789, 9.5808),
    (6, 3, 76, 9.6385, 9.6270),
    (4, 6, 84, 10.0084, 9.9738),
    (5, 5, 84, 9.5270, 9.5267),
    (6, 4, 84, 9.4828, 9.4822),
    (5, 6, 92, 9.5403, 9.5256),
    (6, 5, 92, 9.4551, 9.4505),
    (6, 6, 100, 9.4552, 9.4381),
]

FP16_BASELINE = 9.3895


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
               marker="o", facecolors="none", edgecolors="black", linewidths=1.2,
               s=45, label="path B (table form, hw)", zorder=3)

    for p in points:
        weight = "bold" if p["label"] in frontier_b_labels else "normal"
        ax.annotate(p["label"], (p["bytes"], p["B"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8,
                    fontweight=weight)

    ax.plot([p["bytes"] for p in frontier_b], [p["B"] for p in frontier_b],
            color="black", linewidth=1, linestyle="--", zorder=1,
            label="Pareto frontier (path B)")

    ax.set_xlabel("KV cache bytes / token / layer (8 KV heads)")
    ax.set_ylabel("Perplexity (WikiText-2, mean of 12 windows, 16 layers)")
    # The title used to say "25 key/value widths", but `best_per_bytes` has
    # already collapsed the grid to one point per row size -- 9 of them.  The
    # 25 are all in the asymmetry plot; here they would stack invisibly.
    ax.set_title("Llama 3.2 1B: perplexity vs. KV cache footprint, "
                 "best of 25 key/value widths at each row size")
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
