"""The two architectures, checked against exact arithmetic.

Each is a different order of operations over the same stored codes, so each
has its own specification and each can be wrong in its own way. These
checks are in two layers:

  * every numpy implementation must equal its own specification recomputed in
    exact Python integers -- this catches an int64 overflow, a wrong shift, a
    transposed axis;
  * every path must land within its own truncation bound of the exact real
    inner product -- this catches a structural error, such as scoring in the
    wrong basis, which a self-consistency check would happily confirm.

The fused path is the only one with hardware behind it, so it gets the
strongest form: it must equal the exact integer expression with no slack at
all.
"""

import numpy as np

from kernel.numerics.fixed import INT
from .harness import exact


def _fixture(ctx=96, seed=3, key_bits=4, value_bits=2):
    from kernel.hw.vectors import build
    vs, kernel = build(ctx=ctx, seed=seed, key_bits=key_bits, value_bits=value_bits)
    m = kernel.cfg.model
    kvh = 0
    heads = slice(0, m.kv_groups)
    kv = kernel.cache.view(kvh).select(slice(0, kernel.cache.length))
    return (kernel, vs,
            vs.rotated["q"][0][heads].astype(INT),   # rotated query
            vs.wire["q"][0][heads].astype(INT),      # pre-rotation query
            kv)


def _py(a):
    """numpy -> nested Python ints, so the reference cannot overflow."""
    return np.asarray(a).tolist()


def check_fused_equals_exact_integer_arithmetic():
    """B must equal `(sum q.c) * norm >> 24` computed in unbounded integers.

    `dot` reaches about 2**47 and the norm is 16 bits, so the product sits just
    under the int64 ceiling. If it ever crosses, numpy wraps silently and the
    scores stay plausible -- which is exactly the failure this rules out.
    """
    kernel, _, q_rot, _, kv = _fixture()
    attn = kernel.attn
    attn.score_mode = "fused"
    got, _ = attn.scores(q_rot, kv)

    cent = _py(attn.z.key.codebook.centroids)
    codes = _py(kv.k_idx)
    norms = _py(kv.k_norm)
    qs = _py(q_rot)
    shift = attn.score_shift

    want = []
    for h in qs:
        row = []
        for t, (ct, nt) in enumerate(zip(codes, norms)):
            dot = sum(h[j] * cent[ct[j]] for j in range(len(h)))
            row.append((dot * nt) >> shift)
        want.append(row)
    exact(got, np.array(want, dtype=INT), "fused path disagrees with exact integers")


def check_dense_equals_exact_integer_arithmetic():
    """A: rebuild, rotate back into the original basis, dot the unrotated query.

    Also the only check that the pre-rotation query reaches A with the right
    head slice. A wrong slice would still produce numbers.
    """
    kernel, _, q_rot, q_wire, kv = _fixture()
    attn = kernel.attn
    attn.score_mode = "dense"
    attn._q_unrot = q_wire
    got, _ = attn.scores(q_rot, kv)

    fmt = kernel.cfg.fmt
    dense_shift = attn.z.key.threshold_frac - fmt.qk_frac
    dot_shift = 2 * fmt.qk_frac - fmt.acc_frac
    cent = attn.z.key.codebook.centroids

    k_dense = np.array(
        [[(int(cent[c]) * int(n)) >> dense_shift for c in row]
         for row, n in zip(_py(kv.k_idx), _py(kv.k_norm))], dtype=INT)
    k_orig = _py(kernel.rotation.inverse_apply(k_dense))
    qs = _py(q_wire)

    want = [[sum(h[j] * k[j] for j in range(len(h))) >> dot_shift for k in k_orig]
            for h in qs]
    exact(got, np.array(want, dtype=INT), "A disagrees with exact integers")
    attn.score_mode = "fused"


def check_every_path_approximates_the_same_inner_product():
    """The structural check: same real value, differing only by truncation.

    Self-consistency cannot catch a path that scores in the wrong basis or at
    the wrong scale -- it would agree with its own wrong specification. So each
    path is compared against the exact real inner product it is supposed to be
    computing, and required to land within a few LSB of it.
    """
    kernel, _, q_rot, q_wire, kv = _fixture()
    attn = kernel.attn
    fmt, quant = kernel.cfg.fmt, kernel.cfg.quant

    # The exact value, in Q(acc_frac), as an unrounded real.
    cent = attn.z.key.codebook.centroids.astype(np.float64) / (1 << fmt.centroid_frac)
    k_real = cent[kv.k_idx.astype(INT)] * (
        kv.k_norm.astype(np.float64) / (1 << quant.norm_frac))[:, None]
    q_real = q_rot.astype(np.float64) / (1 << fmt.qk_frac)
    ideal = (q_real @ k_real.T) * (1 << fmt.acc_frac)

    # Budgets leave room for a regression to be caught without being so loose
    # that any bug passes. A's is wider because it rotates back, and each
    # round of that ends in its own truncation.
    bounds = {"fused": 2.0, "dense": 4.0}
    for mode, bound in bounds.items():
        attn.score_mode = mode
        attn._q_unrot = q_wire
        got, _ = attn.scores(q_rot, kv)
        err = float(np.abs(got.astype(np.float64) - ideal).max())
        if err > bound:
            raise AssertionError(
                f"{mode}: worst score is {err:.2f} LSB from the exact inner "
                f"product, past its {bound} LSB budget -- that is a structural "
                f"error, not truncation")
    attn.score_mode = "fused"


