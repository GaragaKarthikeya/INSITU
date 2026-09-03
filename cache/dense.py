"""fp16 K and V, exactly as a conventional serving stack stores them.

The baseline. It exists so that "compression costs X accuracy and saves Y
bytes" is measured against something in this same code path rather than against
a number from a paper.
"""

from __future__ import annotations

import numpy as np

from ..numerics import fp
from ..trace import Op, Trace, record
from .base import KVCache


class DenseCache(KVCache):
    def __init__(self, n_kv_heads: int, head_dim: int, capacity: int) -> None:
        super().__init__(n_kv_heads, head_dim, capacity)
        self.k = np.zeros((capacity, n_kv_heads, head_dim), dtype=fp.FP16)
        self.v = np.zeros_like(self.k)

    @property
    def bytes_per_token(self) -> int:
        return 2 * 2 * self.head_dim * self.n_kv_heads

    def append(self, k, v, positions, trace: Trace | None = None) -> None:
        k = np.atleast_3d(np.asarray(k))
        v = np.atleast_3d(np.asarray(v))
        n = k.shape[0]
        self._check_room(n)
        self.k[self.length:self.length + n] = fp.to_fp16(k)
        self.v[self.length:self.length + n] = fp.to_fp16(v)
        self.length += n
        record(trace, Op.MEM_WRITE, "memory", n_bytes=n * self.bytes_per_token, cache="dense")

    def view(self, kv_head: int, trace: Trace | None = None):
        n = self.length
        record(trace, Op.MEM_READ, "memory",
               n_bytes=n * self.bytes_per_token // self.n_kv_heads, cache="dense")
        return self.k[:n, kv_head], self.v[:n, kv_head]
