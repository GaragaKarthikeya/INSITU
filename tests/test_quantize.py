"""Encode/decode agreement, the wire format, and the distortion it costs."""

import numpy as np

from kernel.config import FixedFormat, QuantConfig
from kernel.numerics.fixed import Q
from kernel.ops.quantize import KVQuantizer, isqrt
from .harness import exact, raises

FMT = FixedFormat()
QQ = Q(FMT.qk_width, FMT.qk_frac)


def _quantizer(d=64, kb=3, vb=2):
    return KVQuantizer(d, QuantConfig(key_bits=kb, value_bits=vb), FMT)


def check_isqrt_is_exact():
    x = np.array([0, 1, 2, 3, 4, 8, 15, 16, 17, 10 ** 6, 10 ** 12, 2 ** 44, 2 ** 44 - 1])
    r = isqrt(x)
    assert np.all(r * r <= x) and np.all((r + 1) ** 2 > x)
    raises(ValueError, lambda: isqrt(np.array([-1])), "negative")


def check_pack_unpack_is_lossless_at_every_width():
    rng = np.random.default_rng(0)
    for kb in range(1, 7):
        for vb in range(1, 7):
            if (64 * (kb + vb)) % 8:
                continue
            z = _quantizer(kb=kb, vb=vb)
            x = QQ.from_float(rng.standard_normal((11, 64)))
            kv = z.encode(z.rotation.apply(x), z.rotation.apply(x * 2))
            back = z.unpack(z.pack(kv))
            for f in ("k_idx", "k_norm", "v_idx", "v_norm"):
                exact(getattr(kv, f), getattr(back, f), f"{f} at {kb}/{vb} bits")


def check_bytes_per_token_matches_the_buffer():
    """The declared size and the actual packed size cannot disagree."""
    z = _quantizer()
    rng = np.random.default_rng(0)
    x = QQ.from_float(rng.standard_normal((4, 64)))
    packed = z.pack(z.encode(z.rotation.apply(x), z.rotation.apply(x)))
    assert packed.shape[-1] == z.bytes_per_token == 44


def check_encoder_and_decoder_agree_on_the_norm():
    """The encoder must threshold against the wire norm, not the full one.

    If it did not, a channel just inside a decision boundary would encode into
    one bin and decode as if it were in another. Detected by re-encoding the
    decoded vector: the codes must be a fixed point.
    """
    z = _quantizer()
    rng = np.random.default_rng(3)
    x = z.rotation.apply(QQ.from_float(rng.standard_normal((50, 64))))
    kv = z.encode(x, x)
    k_hat = z.decode_plane(kv.k_idx, kv.k_norm, z.key)
    again = z._encode_plane(k_hat, z.key, kv.k_norm)
    exact(again, kv.k_idx, "re-encoding the reconstruction is a fixed point")


def check_all_codes_are_used():
    """A codebook whose extreme bins never fire is wasting a bit."""
    z = _quantizer(kb=3, vb=2)
    rng = np.random.default_rng(0)
    x = z.rotation.apply(QQ.from_float(rng.standard_normal((500, 64))))
    kv = z.encode(x, x)
    assert set(np.unique(kv.k_idx)) == set(range(8)), "key codebook underused"
    assert set(np.unique(kv.v_idx)) == set(range(4)), "value codebook underused"


def check_distortion_tracks_the_codebook_bound():
    """Real distortion must be near the Lloyd-Max bound the codebook promises."""
    rng = np.random.default_rng(0)
    expected = {1: 4.4, 2: 9.3, 3: 14.6, 4: 20.2}
    for bits, bound in expected.items():
        z = _quantizer(kb=bits, vb=2)
        x = z.rotation.apply(QQ.from_float(rng.standard_normal((300, 64))))
        kv = z.encode(x, x)
        hat = z.decode_plane(kv.k_idx, kv.k_norm, z.key)
        a, b = QQ.to_float(x), QQ.to_float(hat)
        snr = 10 * np.log10((a ** 2).mean() / ((a - b) ** 2).mean())
        assert snr > bound - 1.0, f"b={bits}: {snr:.2f} dB, bound {bound}"


def check_zero_vector_survives():
    """A masked or padded position has zero norm and must decode to zero."""
    z = _quantizer()
    zeros = np.zeros((2, 64), dtype=np.int64)
    kv = z.encode(zeros, zeros)
    k, v = z.decode(kv)
    exact(k, zeros, "zero key")
    exact(v, zeros, "zero value")


def check_rejects_a_format_that_would_truncate_the_comparison():
    bad = FixedFormat(qk_frac=40)
    raises(ValueError, lambda: KVQuantizer(64, QuantConfig(), bad),
           "threshold_frac below qk_frac")


def check_shadow_matches_the_packed_buffer():
    """The cache's decoded shadow must never drift from the wire format.

    The shadow exists for speed; the packed bytes are what the byte counts are
    measured off. If the two disagreed, every accuracy number would come from
    codes that no hardware reading that buffer would produce.
    """
    from kernel.cache import CompressedCache
    quant = QuantConfig(key_bits=4, value_bits=2)
    c = CompressedCache(3, 64, 64, quant, FMT)
    z = c.quantizer
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = QQ.from_float(rng.standard_normal((7, 3, 64)))
        c.append_rotated(z.rotation.apply(x), z.rotation.apply(x * 1.5))
    for head in range(3):
        shadow, buf = c.view(head), c.view_from_buffer(head)
        for f in ("k_idx", "k_norm", "v_idx", "v_norm"):
            exact(getattr(shadow, f), getattr(buf, f), f"shadow {f} head {head}")
