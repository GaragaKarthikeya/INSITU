"""The KV cache interface.

One interface, two implementations, chosen at construction:

    DenseCache       fp16 K and V. The baseline everything is measured against.
    CompressedCache  codes and norms in the rotated domain. This project.

They are interchangeable inside `AttentionKernel`, which is the point. Asking
"what does compression cost" is then a constructor argument rather than a
second copy of the kernel with the storage swapped out and everything else
hoped to be identical.

The interface is written for the two things a real serving loop does and the
old harness could not express:

    append()    prefill, many tokens at once
    view()      read the cache for one KV head, without copying it
"""

from __future__ import annotations

import abc

import numpy as np

from ..trace import Trace


class KVCache(abc.ABC):
    """Per-layer, per-KV-head storage for one sequence.

    Capacity is fixed at construction and never grows on its own. A
    reallocation in the middle of a decode loop is a latency spike no memory
    model would predict, so running out of capacity is an error the caller has
    to handle.
    """

    def __init__(self, n_kv_heads: int, head_dim: int, capacity: int) -> None:
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.capacity = capacity
        self.length = 0

    @property
    @abc.abstractmethod
    def bytes_per_token(self) -> int:
        """One token, all KV heads. What the off-die port actually moves."""

    @abc.abstractmethod
    def append(self, k: np.ndarray, v: np.ndarray, positions: np.ndarray,
               trace: Trace | None = None) -> None:
        """Store `(tokens, n_kv_heads, head_dim)` of post-RoPE K and V."""

    @abc.abstractmethod
    def view(self, kv_head: int, trace: Trace | None = None):
        """Everything stored for one KV head, in causal order.

        Causal order is not a convenience. It decides which tokens trigger a
        rescale in the online softmax, and so it decides the exact truncation
        pattern. A cache that returned tokens in any other order would give a
        different answer and still look perfectly correct.
        """

    def reset(self) -> None:
        self.length = 0

    def _check_room(self, n: int) -> None:
        if self.length + n > self.capacity:
            raise ValueError(
                f"cache holds {self.length} of {self.capacity} tokens; cannot "
                f"append {n}. Grow `capacity` at construction -- growing here "
                f"would hide a reallocation inside a decode step."
            )

    def stats(self) -> dict:
        return {
            "tokens": self.length,
            "capacity": self.capacity,
            "bytes_per_token": self.bytes_per_token,
            "bytes_resident": self.length * self.bytes_per_token,
        }
