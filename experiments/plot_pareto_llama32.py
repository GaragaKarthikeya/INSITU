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

try:                                    # `python -m kernel.experiments.<name>`
    from kernel.experiments import palette
except ImportError:                     # running the file directly
    import palette

NUM_KV_HEADS = 8      # Llama 3.2 1B
HEAD_DIM = 64
BEAT_BYTES = 16       # one 128-bit AXI-HP port, per cycle
AXI_HP_PORTS = 4      # what the ZCU104's PS exposes to the PL
SHIPPED = "k4v2"      # the configuration the bitstream was built for

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


def planes(key_bits: int, value_bits: int) -> int:
    """Ports this configuration needs, which is what the board actually limits.

    A plane never straddles a field boundary -- `hw/ddr_layout.py` cuts the row
    at its fields first and only then into beats -- so this is not a byte count.
    3b/3b and 4b/2b are both 52-byte rows; one needs five ports and the other
    four.
    """
    kb = HEAD_DIM * key_bits // 8
    vb = HEAD_DIM * value_bits // 8
    return -(-kb // BEAT_BYTES) + -(-vb // BEAT_BYTES) + 1


def plot(out: Path) -> None:
    points = [
        {"label": f"k{kb}v{vb}", "bytes": row * NUM_KV_HEADS, "A": a, "B": b,
         "planes": planes(kb, vb)}
        for kb, vb, row, a, b in GRID
    ]
    points = best_per_bytes(points)

    frontier_b = pareto_frontier(points, "B")
    frontier_b_labels = {p["label"] for p in frontier_b}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Split by what the board can build. Everything to the right of 4b/2b needs
    # more AXI-HP ports than the ZCU104 has, so the quality still on the table
    # there is quality this board cannot reach -- which is the figure's point as
    # much as the shape of the curve is.
    fits = [p for p in points if p["planes"] <= AXI_HP_PORTS]
    over = [p for p in points if p["planes"] > AXI_HP_PORTS]

    ax.plot([p["bytes"] for p in frontier_b], [p["B"] for p in frontier_b],
            color=palette.BLUE, linewidth=1.4, linestyle="--", zorder=1,
            label="Pareto frontier (path B)")

    ax.scatter([p["bytes"] for p in over], [p["B"] for p in over],
               marker="o", facecolors=palette.GREY_FILL,
               edgecolors=palette.GREY, linewidths=1.3, s=55, zorder=3,
               label=f"needs more than {AXI_HP_PORTS} AXI-HP ports")
    ax.scatter([p["bytes"] for p in fits], [p["B"] for p in fits],
               marker="o", facecolors=palette.BLUE_FILL,
               edgecolors=palette.BLUE, linewidths=1.6, s=70, zorder=4,
               label=f"fits the ZCU104's {AXI_HP_PORTS} AXI-HP ports")

    shipped = next((p for p in points if p["label"] == SHIPPED), None)
    if shipped is not None:
        ax.scatter([shipped["bytes"]], [shipped["B"]], marker="o",
                   facecolors="none", edgecolors=palette.VERMILION,
                   linewidths=2.2, s=190, zorder=5,
                   label=f"{SHIPPED}: the configuration built")

    for p in points:
        weight = "bold" if p["label"] in frontier_b_labels else "normal"
        colour = (palette.VERMILION if p["label"] == SHIPPED
                  else palette.BLUE if p["planes"] <= AXI_HP_PORTS
                  else palette.GREY)
        ax.annotate(p["label"], (p["bytes"], p["B"]),
                    textcoords="offset points", xytext=(7, 6), fontsize=8,
                    fontweight=weight, color=colour)

    ax.set_xlabel("KV cache bytes / token / layer (8 KV heads)")
    ax.set_ylabel("Perplexity (WikiText-2, mean of 12 windows, 16 layers)")
    # Log y: k2v2 is 103 against 9.4 for everything above k3v2, so on a linear
    # axis the one configuration nobody would ship flattens the eight that are
    # actually being chosen between into an unreadable band.
    ax.set_yscale("log")
    ax.set_yticks([10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

    # The title used to say "25 key/value widths", but `best_per_bytes` has
    # already collapsed the grid to one point per row size -- 9 of them.  The
    # 25 are all in the asymmetry plot; here they would stack invisibly.
    ax.set_title("Llama 3.2 1B: perplexity vs. KV cache footprint\n"
                 "(best of 25 key/value widths at each row size)")
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
