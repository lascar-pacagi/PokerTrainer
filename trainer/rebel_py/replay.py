"""Reservoir replay buffer for ReBeL value-net training examples.

Each example is a ``(query, value)`` pair: ``query`` is the PBS encoding
(``pbs.encode_query``, dim ``query_dim``) and ``value`` is the per-hand
counterfactual target (``root_values_means`` for the traverser, dim
``NUM_HANDS``). Both are fixed-length, so unlike the Deep-CFR token buffer this
one is a flat preallocated float32 matrix.

Reservoir (Algorithm R, Vitter 1985): once full, an incoming example replaces a
uniformly random slot with probability ``capacity / n_seen``. This keeps the
buffer an unbiased uniform sample of the entire self-play stream, which the
training objective (regress the net onto the running-mean root values) relies
on — matching the reference ``ReplayBuffer`` semantics in the rela framework.
"""
from __future__ import annotations

import numpy as np


class QueryValueBuffer:
    def __init__(self, capacity: int, query_dim: int, value_dim: int,
                 seed: int = 0):
        self.capacity = capacity
        self.q = np.zeros((capacity, query_dim), dtype=np.float32)
        self.v = np.zeros((capacity, value_dim), dtype=np.float32)
        self.n_seen = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, query: np.ndarray, value: np.ndarray) -> None:
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
        else:
            j = int(self.rng.integers(0, self.n_seen + 1))
            if j >= self.capacity:
                self.n_seen += 1
                return
            idx = j
        self.q[idx] = query
        self.v[idx] = value
        self.n_seen += 1

    def add_batch(self, queries: np.ndarray, values: np.ndarray) -> None:
        for query, value in zip(queries, values):
            self.add(query, value)

    def sample(self, batch: int):
        n = min(batch, self.size)
        idx = self.rng.integers(0, self.size, size=n)
        return self.q[idx], self.v[idx]

    def __len__(self) -> int:
        return self.size
