"""Rotate-then-quantize: a dense KV vector in, codes and a norm out.

THE ARITHMETIC, AND WHY THERE IS NO DIVISION
--------------------------------------------
The obvious encoder normalises (`x / norm`) and then compares against the
codebook's decision boundaries. That needs a divide per channel, which is the
most expensive thing in the datapath and buys nothing: comparing `x / norm`
against `b` is the same test as comparing `x` against `b * norm`, and the
second is a multiply. So the boundaries are scaled by the norm once per vector
and every channel is a plain comparison.

THE NORM IS QUANTISED BEFORE IT IS USED, NOT AFTER
--------------------------------------------------
The decoder only ever sees the norm at its wire precision. If the encoder
thresholded against the full-precision norm, encoder and decoder would be
working from different numbers and a channel near a boundary would decode into
the wrong bin. So the norm is rounded to the wire format first, and the
thresholds are built from the rounded value. This is the kind of mismatch that
costs a fraction of a percent of accuracy and is invisible in every test that
does not specifically look for it.

Rounding of the norm is half-to-even. Norms are always positive, so a floor
would bias every single one of them downward, and that bias accumulates across
an entire cache rather than averaging out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FixedFormat, QuantConfig
from ..numerics.fixed import INT, Q, lshift, rshift
from ..trace import Op, Trace, record
from .codebook import Codebook
from .rotate import Rotation


def isqrt(x) -> np.ndarray:
    """Exact integer square root, vectorised.

    float64's 53-bit mantissa covers the operand range here, but "covers" is
    not "is exact", so the float result seeds a correction that is checked
    rather than trusted. Two candidates either side is always enough: the float
    seed is never off by more than one.
    """
    a = np.asarray(x, dtype=INT)
    if np.any(a < 0):
        raise ValueError("isqrt of a negative value")
    r = np.sqrt(a.astype(np.float64)).astype(INT)
    r = np.maximum(r - 1, 0)
    for _ in range(3):
        r = np.where((r + 1) * (r + 1) <= a, r + 1, r)
    if not np.all((r * r <= a) & ((r + 1) * (r + 1) > a)):
        raise AssertionError("isqrt correction failed; operand out of float64 range")
    return r


@dataclass(frozen=True)
class Plane:
    """One quantised field of a token: a codebook plus the format it lands in.

    Keys and values are separate planes because they are quantised at different
    widths and are consumed by different stages -- a key is only ever dotted
    with the query, a value is only ever scaled by a probability. Nothing is
    gained by forcing them to share a width, and a lot of accuracy is lost.
    """

    name: str
    bits: int
    codebook: Codebook
    d: int
    qk_frac: int
    norm_frac: int
    norm_q: Q

    @property
    def centroid_frac(self) -> int:
        return self.codebook.frac

    @property
    def threshold_frac(self) -> int:
        """Q-format the comparison happens in: `boundary * norm`."""
        return self.centroid_frac + self.norm_frac

    @property
    def code_bytes(self) -> int:
        total = self.d * self.bits
        if total % 8:
            raise ValueError(f"plane {self.name}: {self.d}*{self.bits} bits is not whole bytes")
        return total // 8


@dataclass(frozen=True)
class CompressedKV:
    """One token's compressed key and value, for one KV head.

    Arrays are shaped `(..., d)` and `(...,)` so a whole prefill quantises in
    one call and a decode step is just the batch-of-one case. There is no
    separate scalar path to keep in agreement.
    """

    k_idx: np.ndarray     # (..., d) uint8 codes
    k_norm: np.ndarray    # (...,)   int64, Q(norm_frac)
    v_idx: np.ndarray
    v_norm: np.ndarray

    @property
    def n_tokens(self) -> int:
        return int(np.prod(self.k_idx.shape[:-1])) if self.k_idx.ndim > 1 else 1

    def select(self, sl) -> "CompressedKV":
        return CompressedKV(self.k_idx[sl], self.k_norm[sl], self.v_idx[sl], self.v_norm[sl])


class KVQuantizer:
    """Rotation + two codebooks + the wire format, for one head dimension.

    Constructed from a `QuantConfig` and a `FixedFormat` and then immutable, so
    it can be shared across every layer and every KV head without a copy. The
    rotation seed is per-instance, which is what lets two layers use different
    sign diagonals if that is ever wanted, without a global.
    """

    def __init__(self, d: int, quant: QuantConfig, fmt: FixedFormat) -> None:
        self.d = d
        self.quant = quant
        self.fmt = fmt
        self.rotation = Rotation.for_quant(quant, d)

        norm_q = Q(quant.norm_bits, quant.norm_frac)
        self.key = Plane(
            "key", quant.key_bits,
            Codebook.build(quant.codebook, quant.key_bits, fmt.centroid_frac,
                           quant.symmetric),
            d, fmt.qk_frac, quant.norm_frac, norm_q,
        )
        self.value = Plane(
            "value", quant.value_bits,
            Codebook.build(quant.codebook, quant.value_bits, fmt.centroid_frac,
                           quant.symmetric),
            d, fmt.qk_frac, quant.norm_frac, norm_q,
        )

        # The comparison is exact only if scaling x up loses nothing.
        if self.key.threshold_frac < fmt.qk_frac:
            raise ValueError(
                f"centroid_frac + norm_frac = {self.key.threshold_frac} is below "
                f"qk_frac = {fmt.qk_frac}; the threshold comparison would have to "
                f"truncate x, which changes which bin a channel lands in"
            )

    @property
    def bytes_per_token(self) -> int:
        return self.key.code_bytes + self.value.code_bytes + 2 * (self.quant.norm_bits // 8)

    # -- encode ------------------------------------------------------------

    def _norm_wire(self, x_rot: np.ndarray) -> np.ndarray:
        """RMS of the rotated vector, at wire precision.

        RMS rather than L2 so the scale is independent of `d`: `x / rms` has
        unit variance per channel, which is exactly the distribution
        `Codebook.gaussian` is the optimal quantizer for.
        """
        sq = (x_rot.astype(INT) ** 2).sum(axis=-1)      # Q(2 * qk_frac)
        mean_sq = sq // INT(self.d)                      # still Q(2 * qk_frac)
        norm = isqrt(mean_sq)                            # Q(qk_frac)
        wire = self.key.norm_q.requantize(
            norm, src_frac=self.fmt.qk_frac, rounding="even"
        )
        # A zero norm makes every threshold zero and every code the same. It
        # is a real input (a masked or padded position), so it decodes to the
        # zero vector rather than raising -- but it must not be confused with
        # a norm that merely underflowed the wire format.
        return wire

    def _encode_plane(self, x_rot: np.ndarray, plane: Plane,
                      norm_wire: np.ndarray) -> np.ndarray:
        # thresholds: (..., 1, 1) * (2**b - 1,) -> (..., 1, 2**b - 1), one extra
        # axis so it broadcasts against x_scaled's per-channel axis rather than
        # against the token axis, which has the same length often enough that
        # the mistake would survive a small test.
        thresholds = norm_wire[..., None, None].astype(INT) * plane.codebook.boundaries
        x_scaled = lshift(x_rot, plane.threshold_frac - plane.qk_frac)
        # `>` and not `>=`: a value exactly on a boundary falls to the LOWER
        # bin, matching the midpoint convention `Codebook.gaussian` builds the
        # boundaries under. The two must agree or every tie shifts by one code.
        idx = (x_scaled[..., None] > thresholds).sum(axis=-1)
        return idx.astype(np.uint8)

    def encode(self, k_rot: np.ndarray, v_rot: np.ndarray,
               trace: Trace | None = None) -> CompressedKV:
        """Already-rotated K and V in Q(qk_frac) -> codes and norms."""
        k_norm = self._norm_wire(k_rot)
        v_norm = self._norm_wire(v_rot)
        out = CompressedKV(
            k_idx=self._encode_plane(k_rot, self.key, k_norm),
            k_norm=k_norm,
            v_idx=self._encode_plane(v_rot, self.value, v_norm),
            v_norm=v_norm,
        )
        n = out.n_tokens
        record(trace, Op.QUANTIZE, "encoder", m=self.d, k=1, n=2 * n,
               n_bytes=n * self.bytes_per_token)
        return out

    # -- decode ------------------------------------------------------------

    def decode_plane(self, idx: np.ndarray, norm: np.ndarray, plane: Plane) -> np.ndarray:
        """Codes and norm -> the reconstructed vector in Q(qk_frac)."""
        c = plane.codebook.centroids[idx.astype(INT)]           # Q(centroid_frac)
        prod = c * norm[..., None].astype(INT)                   # Q(threshold_frac)
        return rshift(prod, plane.threshold_frac - plane.qk_frac)

    def decode(self, kv: CompressedKV, trace: Trace | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """The full dequantisation. Arm-B's path, and the accuracy reference.

        The compressed attention path never calls this -- that is the whole
        point of it -- but a dense baseline does, and so does every test that
        asks what the compression cost.
        """
        k = self.decode_plane(kv.k_idx, kv.k_norm, self.key)
        v = self.decode_plane(kv.v_idx, kv.v_norm, self.value)
        record(trace, Op.DEQUANTIZE, "codec", m=self.d, k=1, n=2 * kv.n_tokens)
        return k, v

    # -- wire format -------------------------------------------------------

    def pack(self, kv: CompressedKV) -> np.ndarray:
        """One token -> `bytes_per_token` bytes. Shape (..., bytes_per_token) uint8.

        Codes little-endian LSB-first within the stream, then the two norms as
        little-endian words. Written against one layout description that
        `unpack` also reads, so the two cannot drift: a mismatched bit offset
        decodes every token as garbage of exactly the right size, which looks
        like a datapath bug and is not one.
        """
        parts = [
            _pack_codes(kv.k_idx, self.key.bits),
            _pack_codes(kv.v_idx, self.value.bits),
            _pack_word(kv.k_norm, self.quant.norm_bits),
            _pack_word(kv.v_norm, self.quant.norm_bits),
        ]
        return np.concatenate(parts, axis=-1)

    def unpack(self, buf: np.ndarray) -> CompressedKV:
        nb = self.quant.norm_bits // 8
        kb, vb = self.key.code_bytes, self.value.code_bytes
        if buf.shape[-1] != self.bytes_per_token:
            raise ValueError(f"expected {self.bytes_per_token} B per token, got {buf.shape[-1]}")
        o = 0
        k_idx = _unpack_codes(buf[..., o:o + kb], self.key.bits, self.d); o += kb
        v_idx = _unpack_codes(buf[..., o:o + vb], self.value.bits, self.d); o += vb
        k_norm = _unpack_word(buf[..., o:o + nb], self.quant.norm_bits); o += nb
        v_norm = _unpack_word(buf[..., o:o + nb], self.quant.norm_bits)
        return CompressedKV(k_idx, k_norm, v_idx, v_norm)


# --------------------------------------------------------------------------
# bit packing
# --------------------------------------------------------------------------

def _pack_codes(idx: np.ndarray, bits: int) -> np.ndarray:
    flat = idx.astype(np.uint8).reshape(-1, idx.shape[-1])
    bit = np.unpackbits(flat[:, :, None], axis=2, bitorder="little")[:, :, :bits]
    return np.packbits(bit.reshape(flat.shape[0], -1), axis=1, bitorder="little") \
             .reshape(*idx.shape[:-1], -1)


def _unpack_codes(buf: np.ndarray, bits: int, d: int) -> np.ndarray:
    flat = buf.astype(np.uint8).reshape(-1, buf.shape[-1])
    bit = np.unpackbits(flat, axis=1, bitorder="little")[:, :d * bits].reshape(-1, d, bits)
    pad = np.zeros((*bit.shape[:2], 8 - bits), dtype=np.uint8)
    packed = np.packbits(np.concatenate([bit, pad], axis=2), axis=2, bitorder="little")
    return packed[..., 0].reshape(*buf.shape[:-1], d)


def _pack_word(x: np.ndarray, bits: int) -> np.ndarray:
    nb = bits // 8
    a = np.asarray(x, dtype=INT) & ((INT(1) << bits) - 1)
    return np.stack([(a >> (8 * i)) & 0xFF for i in range(nb)], axis=-1).astype(np.uint8)


def _unpack_word(buf: np.ndarray, bits: int) -> np.ndarray:
    nb = bits // 8
    v = sum(buf[..., i].astype(INT) << (8 * i) for i in range(nb))
    sign = INT(1) << (bits - 1)
    return (v ^ sign) - sign          # sign-extend
