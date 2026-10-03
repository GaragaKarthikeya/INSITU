"""At fixed key_bits + value_bits, does k > v beat k < v? Llama 3.2 1B.

x-axis is the bit budget `key_bits + value_bits` (not bytes -- several splits
share a budget, e.g. (4,2) and (2,4) both sum to 6); y-axis is perplexity,
path B (table form, what the hardware implements). Every (key_bits,
value_bits) combination at a given sum is plotted, color-coded by whether
keys or values got the larger share, so the more-key-bits-than-value-bits
advantage is visible directly rather than asserted.

Source: same 25-config grid as plot_pareto_llama32.py. Every perplexity is the
mean over 12 WikiText-2 windows, superseding the earlier single-run grid
(baseline 7.1439) that plan.MD records; the k > v result holds at every bit
budget under both, with no exceptions.

    python -m kernel.experiments.plot_kv_asymmetry \\
        --out experiments/results/kv_asymmetry_llama32.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

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


def relation(kb: int, vb: int) -> str:
    if kb > vb:
        return "k > v"
    if kb < vb:
        return "k < v"
    return "k = v"


OUTLIER_PPL = 20  # 2b-key configs (66-94.5) are a different regime; drop them


def spread_labels(ys: list[float], min_gap: float, iters: int = 500) -> list[float]:
    """Nudge a list of y-values apart so consecutive ones are >= min_gap,
    splitting each adjustment between both neighbors so a label stays close
    to its own point instead of drifting away from the whole column (which
    anchoring at the lowest value and stacking upward does)."""
    z = sorted(ys)
    n = len(z)
    for _ in range(iters):
        moved = False
        for i in range(n - 1):
            gap = z[i + 1] - z[i]
            if gap < min_gap:
                delta = (min_gap - gap) / 2
                z[i] -= delta
                z[i + 1] += delta
                moved = True
        if not moved:
            break
    order = sorted(range(n), key=lambda i: ys[i])
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = z[rank]
    return out


def plot(out: Path, invert: bool = False) -> None:
    points = [
        {"label": f"({kb},{vb})", "sum": kb + vb, "ppl": b, "rel": relation(kb, vb)}
        for kb, vb, _row, _a, b in GRID
        if b <= OUTLIER_PPL
    ]
    for p in points:
        p["y"] = 1.0 / p["ppl"] if invert else p["ppl"]

    # Monochrome: distinguish by marker shape + fill/edge instead of hue.
    color = {"k > v": "black", "k < v": "black", "k = v": "black"}
    marker = {"k > v": "o", "k < v": "^", "k = v": "s"}
    facecolor = {"k > v": "black", "k < v": "none", "k = v": "0.6"}

    fig, ax = plt.subplots(figsize=(9, 9))

    for rel in ("k > v", "k < v", "k = v"):
        pts = [p for p in points if p["rel"] == rel]
        ax.scatter([p["sum"] for p in pts], [p["y"] for p in pts],
                   marker=marker[rel], edgecolors=color[rel],
                   facecolors=facecolor[rel], linewidths=1.2,
                   s=55, label=rel, zorder=3)

    # Give each label its own vertical slot, spread far enough apart to read
    # without a leader line, instead of sitting exactly on a crowded marker.
    lo = min(p["y"] for p in points)
    hi = max(p["y"] for p in points)
    min_gap = (hi - lo) * 0.025
    sums = sorted({p["sum"] for p in points})
    for s in sums:
        col = sorted((p for p in points if p["sum"] == s), key=lambda p: p["y"])
        label_ys = spread_labels([p["y"] for p in col], min_gap)
        for p, ly in zip(col, label_ys):
            ax.annotate(p["label"], xy=(p["sum"], p["y"]),
                        xytext=(p["sum"] + 0.12, ly), fontsize=8, ha="left",
                        va="center")

    y_label = "1 / Perplexity" if invert else "Perplexity"
    ax.set_xlabel("key_bits + value_bits")
    ax.set_ylabel(f"{y_label}, path B (WikiText-2, 2,048 tok, 16 layers)")
    title_dir = "higher is better" if invert else "lower is better"
    ax.set_title("Llama 3.2 1B: more key bits than value bits wins at every fixed budget\n"
                 f"({title_dir}; 2b-key configs omitted)")
    ax.set_xticks(sums)
    ax.set_xlim(min(sums) - 0.5, max(sums) + 1.3)
    ax.set_ylim(lo - min_gap, hi + min_gap * 6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    print(f"\nBy bit budget (key_bits + value_bits), best to worst ({y_label}):")
    sums = sorted({p["sum"] for p in points})
    for s in sums:
        row = sorted((p for p in points if p["sum"] == s),
                     key=lambda p: -p["y"] if invert else p["y"])
        print(f"  sum={s}: " + ", ".join(f"{p['label']}={p['y']:.4f}" for p in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                     default=Path("experiments/results/kv_asymmetry_llama32.png"))
    ap.add_argument("--invert", action="store_true",
                     help="plot 1/perplexity (higher is better) instead of perplexity")
    args = ap.parse_args()
    plot(args.out, invert=args.invert)


if __name__ == "__main__":
    main()
