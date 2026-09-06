"""The assembled block: shapes, causality, the fold, and the invariants."""

import numpy as np

from kernel import (AttentionKernel, HardwareConfig, KernelConfig, ModelConfig,
                    Op, QuantConfig, Trace, Weights)
from kernel.ops.output import fold_o_proj
from kernel.ops.rotate import Rotation
from .harness import approx, exact, raises

SMALL = ModelConfig(hidden_size=128, num_heads=4, num_kv_heads=2, head_dim=32,
                    max_position=256)


def _kernel(**kw):
    cfg = KernelConfig(model=SMALL, quant=QuantConfig(key_bits=3, value_bits=2))
    return AttentionKernel(cfg, capacity=128, **kw)


def check_shapes_and_dtype():
    k = _kernel()
    rng = np.random.default_rng(0)
    y, rep = k.forward(rng.standard_normal((5, 128)).astype(np.float32))
    assert y.shape == (5, 128) and rep.context == 5
    y2, rep2 = k.forward(rng.standard_normal(128).astype(np.float32))
    assert y2.shape == (1, 128) and rep2.context == 6


def check_prefill_then_decode_equals_one_prefill():
    """Splitting a sequence across calls must not change the answer.

    This is the property a serving loop depends on and the one an
    accelerator model that only ever ran a steady-state decode cannot check.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal((9, 128)).astype(np.float32)

    a = _kernel()
    whole, _ = a.forward(x)

    b = _kernel()
    parts = [b.forward(x[:4])[0], b.forward(x[4:7])[0], b.forward(x[7:])[0]]
    # not bit-identical, and it cannot be: the fp32 GEMM in the projection
    # blocks differently for a batch of 9 than for batches of 4, 3 and 2, so
    # its accumulation order changes. Everything downstream of the fixed-point
    # boundary is exact -- the residual here is fp32 reassociation and nothing
    # else, which is why the bound is at the fp32 epsilon and not merely small.
    approx(np.concatenate(parts), whole, tol=1e-5, what="split vs whole")


def check_causality():
    """Appending later tokens must not change an earlier token's output."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 128)).astype(np.float32)
    a = _kernel()
    first, _ = a.forward(x[:3])
    a.forward(x[3:])
    b = _kernel()
    again, _ = b.forward(x[:3])
    exact(first, again, "earlier outputs are fixed")


def check_reset_restores_a_fresh_sequence():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    k = _kernel()
    a, _ = k.forward(x)
    k.reset()
    b, _ = k.forward(x)
    exact(a, b, "reset")


def check_hardware_config_cannot_change_a_value():
    """The package's central invariant, asserted directly.

    Swapping the hardware model changes only the reduction chunk, so the two
    runs may differ in the last bits of the fp32 projection -- but never
    materially, and never in the fixed-point path at all.
    """
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    w = Weights.random(SMALL, Rotation.from_seed(32, 0), seed=0)

    base = KernelConfig(model=SMALL)
    a = AttentionKernel(base, w, capacity=64)
    b = AttentionKernel(base.with_hardware(HardwareConfig.sized_for(SMALL, side=16)),
                        w, capacity=64)
    ya, _ = a.forward(x)
    yb, _ = b.forward(x)
    rel = np.abs(ya - yb).max() / max(np.abs(ya).max(), 1e-9)
    assert rel < 1e-3, f"hardware config moved the answer by {rel:.2e}"


def check_fold_is_exact_in_real_arithmetic():
    """`W_o'` must reproduce `W_o` applied after an explicit inverse rotation."""
    rng = np.random.default_rng(5)
    r = Rotation.from_seed(SMALL.head_dim, 0)
    w_o = rng.standard_normal((SMALL.hidden_size, SMALL.q_out_features))
    folded = fold_o_proj(w_o, r, SMALL)

    acc = rng.standard_normal((3, SMALL.num_heads, SMALL.head_dim))
    # The path the design avoids: rotate each head back, then project.
    # `apply` acts as `acc = v @ R.T`, so the inverse on a row vector is `@ R`.
    unrotated = np.einsum("thd,de->the", acc, r.matrix())
    explicit = unrotated.reshape(3, -1) @ w_o.T
    approx(acc.reshape(3, -1) @ folded.T, explicit, tol=1e-9, what="fold")


def check_fold_is_not_its_own_transpose():
    """Guards the direction. Both orientations are orthonormal and both run."""
    r = Rotation.from_seed(SMALL.head_dim, 0)
    w_o = np.random.default_rng(0).standard_normal(
        (SMALL.hidden_size, SMALL.q_out_features))
    folded = fold_o_proj(w_o, r, SMALL)
    assert np.abs(folded - w_o).max() > 1e-6, "the fold did nothing"
    # The wrong orientation differs materially, so a transposed fold cannot
    # pass check_fold_is_exact_in_real_arithmetic by accident.
    block = np.zeros((SMALL.q_out_features,) * 2)
    for h in range(SMALL.num_heads):
        s = h * SMALL.head_dim
        block[s:s + SMALL.head_dim, s:s + SMALL.head_dim] = r.matrix()
    assert np.abs(folded - w_o @ block).max() > 1e-6


def check_fold_rejects_a_mismatched_layout():
    r = Rotation.from_seed(SMALL.head_dim, 0)
    bad = np.zeros((SMALL.hidden_size, SMALL.q_out_features + SMALL.head_dim))
    raises(ValueError, lambda: fold_o_proj(bad, r, SMALL), "wrong o_proj width")


def check_compressed_and_dense_agree_to_the_quantisation_error():
    """The A/B the package exists for, with everything but storage held fixed."""
    rng = np.random.default_rng(6)
    x = rng.standard_normal((12, 128)).astype(np.float32)
    w = Weights.random(SMALL, Rotation.from_seed(32, 0), seed=0)
    cfg = KernelConfig(model=SMALL, quant=QuantConfig(key_bits=4, value_bits=4))

    yc, _ = AttentionKernel(cfg, w, capacity=64, compressed=True).forward(x)
    yd, _ = AttentionKernel(cfg, w, capacity=64, compressed=False).forward(x)
    rel = np.abs(yc - yd).max() / np.abs(yd).max()
    assert rel < 0.25, f"4-bit compression moved the block output by {rel:.1%}"
    assert rel > 1e-6, "compressed and dense should NOT be identical"


def check_cache_capacity_is_enforced():
    k = _kernel()
    rng = np.random.default_rng(7)
    raises(ValueError,
           lambda: k.forward(rng.standard_normal((200, 128)).astype(np.float32)),
           "capacity")


def check_position_validation():
    k = _kernel()
    rng = np.random.default_rng(8)
    x = rng.standard_normal((3, 128)).astype(np.float32)
    raises(ValueError, lambda: k.forward(x, positions=np.array([0, 1])), "wrong count")


def check_trace_bytes_match_the_cache():
    """bytes/token is read off the port, never declared."""
    k = _kernel()
    rng = np.random.default_rng(9)
    t = Trace()
    k.forward(rng.standard_normal((10, 128)).astype(np.float32), trace=t)
    written = sum(w.n_bytes for w in t.of(Op.MEM_WRITE)
                  if w.meta.get("cache") == "compressed")
    assert written == 10 * k.cache.bytes_per_token, (
        f"trace says {written} B, cache says {10 * k.cache.bytes_per_token} B")


def check_batching_does_not_change_the_trace():
    """The evaluation strategy must be invisible to the modelled hardware.

    This is the invariant that lets prefill be fast without any claim about
    the design changing. If it ever fails, the speedup has silently become a
    query-tiling architecture -- which is a real and interesting design, but a
    different one, and it must not arrive by accident.
    """
    from collections import Counter
    rng = np.random.default_rng(0)
    x = rng.standard_normal((24, 128)).astype(np.float32)
    w = Weights.random(SMALL, Rotation.from_seed(SMALL.head_dim, 0), seed=0)
    cfg = KernelConfig(model=SMALL, quant=QuantConfig(key_bits=4, value_bits=2))

    def run(force):
        k = AttentionKernel(cfg, w, capacity=64)
        k.force_token_loop = force
        t = Trace()
        y, rep = k.forward(x, trace=t)
        work = Counter()
        for rec in t.records:
            work[rec.op.value] += rec.n
        reads = [rec.n_bytes for rec in t.of(Op.MEM_READ)]
        return y, work, reads, rep

    fast, w_fast, r_fast, rep_f = run(False)
    slow, w_slow, r_slow, rep_s = run(True)
    exact(fast, slow, "batched vs token-at-a-time output")
    assert w_fast == w_slow, f"op work differs: {w_fast} vs {w_slow}"
    assert r_fast == r_slow, "MEM_READ byte counts differ"
    assert rep_f.attention.score_overflows == rep_s.attention.score_overflows
    assert rep_f.attention.acc_overflows == rep_s.attention.acc_overflows


def check_decode_still_uses_the_reference_path():
    """n == 1 must not silently take a different route than a real decode step."""
    rng = np.random.default_rng(1)
    k = _kernel()
    k.forward(rng.standard_normal((8, 128)).astype(np.float32))
    t = Trace()
    k.forward(rng.standard_normal(128).astype(np.float32), trace=t)
    reads = [w for w in t.of(Op.MEM_READ) if w.meta.get("cache") == "compressed"]
    assert len(reads) == SMALL.num_kv_heads, (
        f"a decode step must read each KV head exactly once, got {len(reads)}")


def check_dense_batching_preserves_the_trace():
    """The fp16 control's batched path is an evaluation strategy too.

    Values are checked to a tolerance rather than exactly: BLAS blocks a
    batched GEMM differently from a sequence of GEMVs, so the float64 scores
    differ in their last bits. The trace must still be identical -- that is the
    part that describes hardware.
    """
    from collections import Counter
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 128)).astype(np.float32)
    w = Weights.random(SMALL, Rotation.from_seed(SMALL.head_dim, 0), seed=0)
    cfg = KernelConfig(model=SMALL)

    def run(force):
        k = AttentionKernel(cfg, w, capacity=64, compressed=False)
        k.force_token_loop = force
        t = Trace()
        y, _ = k.forward(x, trace=t)
        work = Counter()
        for rec in t.records:
            work[rec.op.value] += rec.n
        return y, work, [rec.n_bytes for rec in t.of(Op.MEM_READ)]

    fast, w_fast, r_fast = run(False)
    slow, w_slow, r_slow = run(True)
    assert w_fast == w_slow, f"op work differs: {w_fast} vs {w_slow}"
    assert r_fast == r_slow, "MEM_READ byte counts differ"
    rel = np.abs(fast - slow).max() / max(np.abs(slow).max(), 1)
    assert rel < 1e-4, f"dense batch drifted {rel:.2e} from the loop"


def check_rms_norm_matches_the_reference():
    """QK-norm must be the model's norm, not a reimplementation that is close.

    Checked against torch's own RMSNorm formula in float32, which is what the
    reference upcasts to regardless of the model's dtype.
    """
    from kernel.ops.norm import rms_norm
    rng = np.random.default_rng(0)
    for shape in ((5, 4, 64), (1, 32, 128), (7, 128)):
        x = rng.standard_normal(shape).astype(np.float32)
        g = rng.standard_normal(shape[-1]).astype(np.float32)
        eps = 1e-6
        x32 = x.astype(np.float32)
        var = (x32 ** 2).mean(-1, keepdims=True, dtype=np.float32)
        want = g * (x32 / np.sqrt(var + np.float32(eps)))
        approx(rms_norm(x, g, eps), want, tol=1e-5, what=f"rms_norm {shape}")


def check_rms_norm_is_scale_invariant():
    """Which is why the attention scale must be applied after it, not before."""
    from kernel.ops.norm import rms_norm
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4, 32)).astype(np.float32)
    g = np.ones(32, dtype=np.float32)
    approx(rms_norm(x * 17.0, g, 1e-6), rms_norm(x, g, 1e-6), tol=1e-4,
           what="scale invariance")


def check_qk_norm_changes_the_answer():
    """A guard against wiring it in and having it silently do nothing."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((6, 128)).astype(np.float32)
    r = Rotation.from_seed(SMALL.head_dim, 0)
    base = [rng.standard_normal(s).astype(np.float32) * 0.05 for s in (
        (SMALL.q_out_features, 128), (SMALL.kv_out_features, 128),
        (SMALL.kv_out_features, 128), (128, SMALL.q_out_features))]
    gain = rng.standard_normal(SMALL.head_dim).astype(np.float32) * 0.3 + 1.0

    plain = Weights.prepare(*base, SMALL, r)
    normed = Weights.prepare(*base, SMALL, r, q_norm=gain, k_norm=gain)
    cfg = KernelConfig(model=SMALL)
    a, _ = AttentionKernel(cfg, plain, capacity=32).forward(x)
    b, _ = AttentionKernel(cfg, normed, capacity=32).forward(x)
    assert np.abs(a - b).max() > 1e-3, "QK-norm had no effect; it is not wired in"


def check_qk_norm_shape_and_pairing_are_enforced():
    r = Rotation.from_seed(SMALL.head_dim, 0)
    base = [np.zeros(s, dtype=np.float32) for s in (
        (SMALL.q_out_features, 128), (SMALL.kv_out_features, 128),
        (SMALL.kv_out_features, 128), (128, SMALL.q_out_features))]
    g = np.ones(SMALL.head_dim, dtype=np.float32)
    raises(ValueError, lambda: Weights.prepare(*base, SMALL, r, q_norm=g),
           "one-sided QK-norm")
    raises(ValueError,
           lambda: Weights.prepare(*base, SMALL, r,
                                   q_norm=np.ones(SMALL.hidden_size, dtype=np.float32),
                                   k_norm=g),
           "norm over hidden_size instead of head_dim")


def check_bias_is_applied():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    r = Rotation.from_seed(SMALL.head_dim, 0)
    base = [rng.standard_normal(s).astype(np.float32) * 0.05 for s in (
        (SMALL.q_out_features, 128), (SMALL.kv_out_features, 128),
        (SMALL.kv_out_features, 128), (128, SMALL.q_out_features))]
    cfg = KernelConfig(model=SMALL)
    a, _ = AttentionKernel(cfg, Weights.prepare(*base, SMALL, r), capacity=32).forward(x)
    ob = np.full(SMALL.hidden_size, 0.25, dtype=np.float32)
    b, _ = AttentionKernel(cfg, Weights.prepare(*base, SMALL, r, o_bias=ob),
                           capacity=32).forward(x)
    approx(b - a, np.broadcast_to(ob, b.shape), tol=1e-4, what="o_proj bias")


def check_rotation_rounds_do_not_change_the_dense_output():
    """The rotation is applied and then folded back out, so with quantisation
    OFF the number of rounds must be invisible in the output.

    This is the check that was missing. `rot_rounds` reached the quantizer and
    the kernel but not the offline `W_o` fold, so the block's output sat in a
    different basis from the one `W_o'` was built for -- perplexity 3532
    against a baseline of 9.42, and only for the settings that did not match
    the default, which is why nothing else caught it.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 128)).astype(np.float32)
    base = [rng.standard_normal(s).astype(np.float32) * 0.05 for s in (
        (SMALL.q_out_features, 128), (SMALL.kv_out_features, 128),
        (SMALL.kv_out_features, 128), (128, SMALL.q_out_features))]

    ref = None
    for rounds in (1, 2, 3):
        quant = QuantConfig(rot_rounds=rounds)
        rot = Rotation.for_quant(quant, SMALL.head_dim)
        w = Weights.prepare(*base, SMALL, rot)
        k = AttentionKernel(KernelConfig(model=SMALL, quant=quant), w,
                            capacity=32, compressed=False)
        y, _ = k.forward(x)
        if ref is None:
            ref = y
        else:
            rel = np.abs(y - ref).max() / max(np.abs(ref).max(), 1e-9)
            assert rel < 1e-3, (
                f"rot_rounds={rounds} moved the dense output by {rel:.2e}; the "
                f"fold and the kernel are using different rotations")


def check_every_rotation_is_built_the_same_way():
    """`Rotation.for_quant` must be the only constructor any caller uses.

    Grep-as-a-test, because the failure mode is a call site that silently drops
    a field rather than one that errors.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3] / "kernel"
    offenders = []
    for f in root.rglob("*.py"):
        if f.name in ("rotate.py",) or "tests" in f.parts:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if "Rotation.from_seed(" in line:
                offenders.append(f"{f.relative_to(root)}:{i}")
    assert not offenders, (
        "these build a Rotation directly instead of via Rotation.for_quant, "
        f"which is how rot_rounds got dropped once already: {offenders}")
