"""Two ways to score a compressed key, and what each one costs.

    python -m kernel.experiments.compare_paths --ctx 512

The same cache, the same query, the same rotation. The only thing that changes
is where the arithmetic happens, and the point is that both compute the same
inner product while differing by two orders of magnitude in multiplies.

A  the modular design. Rebuild the dense key from its codes and norm,
   truncating into the datapath's format, then dot it with the query. This is
   what a compressed format asks for if you read it as a specification: a
   decompressor, then a conventional attention unit.

B  what this project builds. Never rebuilds the key. The query's products
   against the codebook are tabulated once per decode step; a cached token
   selects from them, and the norm is applied once after the sum.
"""

from __future__ import annotations

import argparse

import numpy as np

from ..numerics.fixed import INT, rshift


def score_paths(kernel, vs, kv_head: int):
    """Scores for one KV head, by all three routes. Each lands in Q(acc_frac)."""
    fmt, quant = kernel.cfg.fmt, kernel.cfg.quant
    m = kernel.cfg.model
    z = kernel.cache.quantizer
    heads = slice(kv_head * m.kv_groups, (kv_head + 1) * m.kv_groups)

    q_rot = vs.rotated["q"][0][heads].astype(INT)
    kv = kernel.cache.view(kv_head).select(slice(0, kernel.cache.length))

    cent = z.key.codebook.centroids
    codes = cent[kv.k_idx.astype(INT)]               # (T, d) Q(centroid_frac)
    norm = kv.k_norm.astype(INT)[:, None]            # (T, 1) Q(norm_frac)

    # -- the dense key both A paths build, truncated into the datapath format
    dense_shift = z.key.threshold_frac - fmt.qk_frac          # 24 - 16 = 8
    k_dense = rshift(codes * norm, dense_shift)               # (T, d) Q(qk_frac)

    # A: score the rebuilt dense key.
    dot_shift = 2 * fmt.qk_frac - fmt.acc_frac                # 32 - 16
    s_a = rshift(np.einsum("hj,tj->ht", q_rot, k_dense), dot_shift)

    # B: no dense key at all. Sum first, norm once, one shift.
    dot = np.einsum("hj,tj->ht", q_rot, codes)
    s_b = rshift(dot * norm[:, 0], fmt.score_shift_for(quant))

    return s_a, s_b


def multiplies(d: int, key_bits: int, tokens: int, rounds: int, log2d: int):
    """Multiplies to score `tokens` cached keys against one query, per KV head."""
    return {
        # rebuild (d multiplies) then dot (d more), per cached token
        "A": {"per_token_mult": 2 * d, "per_token_addsub": 0, "once": 0},
        # the table is d x 2**bits, built once; a token costs one norm multiply
        "B":  {"per_token_mult": 1, "per_token_addsub": 0,
               "once": d * (1 << key_bits)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key-bits", type=int, default=4)
    ap.add_argument("--value-bits", type=int, default=2)
    a = ap.parse_args(argv)

    from ..hw.vectors import build

    vs, kernel = build(ctx=a.ctx, seed=a.seed,
                       key_bits=a.key_bits, value_bits=a.value_bits)
    m = kernel.cfg.model
    d = m.head_dim
    print(f"d={d}, {m.num_kv_heads} KV heads, {kernel.cache.length} cached tokens, "
          f"{a.key_bits}b keys")
    print()

    diffs, span, n = [], 0, 0
    for h in range(m.num_kv_heads):
        s_a, s_b = score_paths(kernel, vs, h)
        diffs.append(np.abs(s_a - s_b).ravel())
        span = max(span, int(np.abs(s_b).max()))
        n += s_b.size
    err = np.concatenate(diffs)

    print(f"scores compared: {n:,} across all KV heads, |s| up to {span:,}")
    same = int((err == 0).sum())
    print(f"  A vs B:  {same:,} of {n:,} identical ({100.0*same/n:.1f}%),  "
          f"max {int(err.max())} LSB,  mean {err.mean():.4f}")
    print()

    log2d = d.bit_length() - 1
    cost = multiplies(d, a.key_bits, kernel.cache.length,
                      kernel.rotation.rounds, log2d)
    T = kernel.cache.length
    print(f"multiplies to score {T} cached tokens against one query, per KV head:")
    for name in ("A", "B"):
        c = cost[name]
        total = c["once"] + c["per_token_mult"] * T
        extra = (f"  + {c['per_token_addsub'] * T:,} add/sub for the rotations"
                 if c["per_token_addsub"] else "")
        print(f"  {name}: {total:>10,}{extra}")
    ratio = (cost["A"]["per_token_mult"] * T) / (cost["B"]["once"] + T)
    print(f"\n  A/B = {ratio:.1f}x at this context, and it grows with context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
