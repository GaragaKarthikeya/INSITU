"""At fixed key_bits + value_bits, does k > v beat k < v? Llama 3.2 1B.

x-axis is the bit budget `key_bits + value_bits` (not bytes -- several splits
share a budget, e.g. (4,2) and (2,4) both sum to 6); y-axis is perplexity,
path B (table form, what the hardware implements). Every (key_bits,
value_bits) combination at a given sum is plotted, color-coded by whether
keys or values got the larger share, so the more-key-bits-than-value-bits
advantage is visible directly rather than asserted.

Source: same 25-config grid as plot_pareto_llama32.py, from plan.MD /
experiments/results/paths/sweep_AB.log.

    python -m kernel.experiments.plot_kv_asymmetry \\
        --out experiments/results/kv_asymmetry_llama32.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

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


def relation(kb: int, vb: int) -> str:
    if kb > vb:
        return "k > v"
    if kb < vb:
        return "k < v"
    return "k = v"


def plot(out: Path) -> None:
    points = [
        {"label": f"({kb},{vb})", "sum": kb + vb, "ppl": b, "rel": relation(kb, vb)}
        for kb, vb, _row, _a, b in GRID
    ]

    color = {"k > v": "tab:blue", "k < v": "tab:red", "k = v": "tab:green"}
    marker = {"k > v": "o", "k < v": "^", "k = v": "s"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for rel in ("k > v", "k < v", "k = v"):
        pts = [p for p in points if p["rel"] == rel]
        ax.scatter([p["sum"] for p in pts], [p["ppl"] for p in pts],
                   color=color[rel], marker=marker[rel], s=45, label=rel, zorder=3)

    # Labels within the same x-column collide when their ppl values are close,
    # so stagger each column's labels through a small set of offsets instead
    # of one fixed corner -- and draw a thin leader line to the point, since a
    # displaced label is otherwise ambiguous about which marker it names.
    offsets = [(14, 0), (14, 14), (14, -14), (-14, 14), (-14, -14), (-14, 0)]
    sums = sorted({p["sum"] for p in points})
    for s in sums:
        col = sorted((p for p in points if p["sum"] == s), key=lambda p: p["ppl"])
        for i, p in enumerate(col):
            dx, dy = offsets[i % len(offsets)]
            ha = "left" if dx > 0 else "right"
            ax.annotate(p["label"], (p["sum"], p["ppl"]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=7, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", color="gray",
                                         lw=0.5, shrinkA=0, shrinkB=3))

    ax.set_xlabel("key_bits + value_bits")
    ax.set_ylabel("Perplexity, path B (WikiText-2, 2,048 tok, 16 layers)")
    ax.set_title("Llama 3.2 1B: more key bits than value bits wins at every fixed budget")
    ax.set_xticks(sums)
    ax.set_xlim(min(sums) - 0.5, max(sums) + 0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print("\nBy bit budget (key_bits + value_bits), best to worst ppl(B):")
    sums = sorted({p["sum"] for p in points})
    for s in sums:
        row = sorted((p for p in points if p["sum"] == s), key=lambda p: p["ppl"])
        print(f"  sum={s}: " + ", ".join(f"{p['label']}={p['ppl']:.4f}" for p in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                     default=Path("experiments/results/kv_asymmetry_llama32.png"))
    args = ap.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()
