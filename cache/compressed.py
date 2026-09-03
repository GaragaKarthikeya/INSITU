"""Codes and norms in the rotated domain -- the cache this project is about.

THE PACKED BUFFER IS AUTHORITATIVE
----------------------------------
Tokens are stored as PACKED BYTES. The byte count is then a property of the
thing in memory rather than a figure computed in a document, and
`bytes_per_token` is measured off the same buffer the trace charges for.

A DECODED SHADOW SITS BESIDE IT, AND IT IS NOT A SHORTCUT
--------------------------------------------------------
`view` used to bit-unpack the whole cache on every call, and every query token
against every KV head is a call -- 73% of total runtime went into unpacking the
same bytes again and again. That work corresponds to NO HARDWARE: a real design
unpacks a beat once as it streams off the port, not once per consumer.

So the codes are also kept unpacked, written on `append_rotated` from the
`CompressedKV` the encoder already produced. Two things make this safe rather
than convenient:

  * **The trace is untouched.** `view` still records exactly the same MEM_READ
    it always did, with the same byte count. What changed is how the bytes are
    turned into codes in Python, not how many bytes the modelled design moves.
    `tests/test_kernel.py::check_view_records_one_read_per_call` pins this.
  * **The shadow is checked against the buffer**, not trusted.
    `tests/test_quantize.py::check_shadow_matches_the_packed_buffer` asserts
    they are bit-identical, so the shadow cannot drift from the wire format --
    which is the only way this could ever become a lie.
"""

from __future__ import annotations

import numpy as np

from ..config import FixedFormat, QuantConfig
from ..ops.quantize import CompressedKV, KVQuantizer
from ..trace import Op, Trace, record
from .base import KVCache


class CompressedCache(KVCache):
    def __init__(self, n_kv_heads: int, head_dim: int, capacity: int,
                 quant: QuantConfig, fmt: FixedFormat, qk_frac: int | None = None) -> None:
        super().__init__(n_kv_heads, head_dim, capacity)
        self.quantizer = KVQuantizer(head_dim, quant, fmt)
        self.fmt = fmt
        self._row = self.quantizer.bytes_per_token
        self.buf = np.zeros((capacity, n_kv_heads, self._row), dtype=np.uint8)

        # The decoded shadow. Preallocated so `append_rotated` never reallocates
        # mid-sequence, for the same reason `capacity` is fixed: a realloc
        # inside a decode step is a latency spike no model would predict.
        shape = (capacity, n_kv_heads, head_dim)
        self._k_idx = np.zeros(shape, dtype=np.uint8)
        self._v_idx = np.zeros(shape, dtype=np.uint8)
        self._k_norm = np.zeros((capacity, n_kv_heads), dtype=np.int64)
        self._v_norm = np.zeros((capacity, n_kv_heads), dtype=np.int64)

    @property
    def bytes_per_token(self) -> int:
        return self._row * self.n_kv_heads

    def append_rotated(self, k_rot, v_rot, trace: Trace | None = None) -> None:
        """Store K and V that are ALREADY rotated and in Q(qk_frac).

        The rotation belongs to the write path, not to the cache, and the
        kernel does it before it gets here -- fused with nothing, so the same
        rotated tensors also feed the dense baseline when one is running.
        """
        k_rot = np.asarray(k_rot)
        n = k_rot.shape[0]
        self._check_room(n)
        kv = self.quantizer.encode(k_rot, v_rot, trace)

        lo, hi = self.length, self.length + n
        self.buf[lo:hi] = self.quantizer.pack(kv)
        self._k_idx[lo:hi] = kv.k_idx
        self._v_idx[lo:hi] = kv.v_idx
        self._k_norm[lo:hi] = kv.k_norm
        self._v_norm[lo:hi] = kv.v_norm
        self.length = hi

        record(trace, Op.MEM_WRITE, "memory",
               n_bytes=n * self.bytes_per_token, cache="compressed")

    def append(self, k, v, positions, trace: Trace | None = None) -> None:
        raise NotImplementedError(
            "CompressedCache stores rotated tokens; call append_rotated. "
            "Rotating inside the cache would mean the dense baseline and the "
            "compressed path rotate in two different places."
        )

    def view(self, kv_head: int, trace: Trace | None = None,
             n_reads: int = 1) -> CompressedKV:
        """One KV head's cache, in causal order, charging `n_reads` reads for it.

        `n_reads` exists because the modelled design reads this cache once per
        QUERY TOKEN, while Python may serve many query tokens from one call.
        Passing the number of query tokens keeps the trace -- and therefore
        every byte, cycle and energy figure -- identical to the
        token-at-a-time path. It is an accounting parameter, not a hardware
        one: raising it does not batch anything and does not save any traffic.

        The byte count is the WHOLE resident cache, not the causal prefix a
        given query token needs. That is what the model has always charged and
        it is left alone deliberately: for decode -- the case every ns/token
        and energy figure refers to -- resident length and causal prefix are
        the same thing. For prefill it is an overcharge, and changing it would
        be a change to the modelled design, not to this function.
        """
        n = self.length
        for _ in range(n_reads):
            record(trace, Op.MEM_READ, "memory",
                   n_bytes=n * self._row, cache="compressed")
        return CompressedKV(
            k_idx=self._k_idx[:n, kv_head], k_norm=self._k_norm[:n, kv_head],
            v_idx=self._v_idx[:n, kv_head], v_norm=self._v_norm[:n, kv_head],
        )

    def view_from_buffer(self, kv_head: int) -> CompressedKV:
        """The same view, decoded from the packed bytes. For tests only.

        Exists so `view`'s shadow can be checked against the wire format rather
        than assumed to agree with it.
        """
        return self.quantizer.unpack(self.buf[:self.length, kv_head])

    def reset(self) -> None:
        super().reset()
