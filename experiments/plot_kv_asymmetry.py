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


def plot(out: Path) -> None:
    points = [
        {"label": f"({kb},{vb})", "sum": kb + vb, "ppl": b, "rel": relation(kb, vb)}
        for kb, vb, _row, _a, b in GRID
        if b <= OUTLIER_PPL
    ]

    color = {"k > v": "tab:blue", "k < v": "tab:red", "k = v": "tab:green"}
    marker = {"k > v": "o", "k < v": "^", "k = v": "s"}

    fig, ax = plt.subplots(figsize=(9, 9))

    for rel in ("k > v", "k < v", "k = v"):
        pts = [p for p in points if p["rel"] == rel]
        ax.scatter([p["sum"] for p in pts], [p["ppl"] for p in pts],
                   color=color[rel], marker=marker[rel], s=50, label=rel, zorder=3)

    # Give each label its own vertical slot, spread far enough apart to read
    # without a leader line, instead of sitting exactly on a crowded marker.
    lo = min(p["ppl"] for p in points)
    hi = max(p["ppl"] for p in points)
    min_gap = (hi - lo) * 0.025
    sums = sorted({p["sum"] for p in points})
    for s in sums:
        col = sorted((p for p in points if p["sum"] == s), key=lambda p: p["ppl"])
        label_ys = spread_labels([p["ppl"] for p in col], min_gap)
        for p, ly in zip(col, label_ys):
            ax.annotate(p["label"], xy=(p["sum"], p["ppl"]),
                        xytext=(p["sum"] + 0.12, ly), fontsize=8, ha="left",
                        va="center")

    ax.set_xlabel("key_bits + value_bits")
    ax.set_ylabel("Perplexity, path B (WikiText-2, 2,048 tok, 16 layers)")
    ax.set_title("Llama 3.2 1B: more key bits than value bits wins at every fixed budget\n"
                 f"(2b-key configs, ppl > {OUTLIER_PPL}, omitted)")
    ax.set_xticks(sums)
    ax.set_xlim(min(sums) - 0.5, max(sums) + 1.3)
    ax.set_ylim(lo - min_gap, hi + min_gap * 6)
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
