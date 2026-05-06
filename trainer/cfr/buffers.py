"""Reservoir buffers for Deep CFR.

Two buffer roles:

* **AdvantageBuffer** stores `(state x, regret_vec, t)` triples — the input to
  refitting the AdvNet at every CFR iteration. One buffer per traverser
  player (so `len(adv_bufs) == NUM_PLAYERS == 2`).
* **PolicyBuffer** stores `(state x, strategy_vec, t)` — the input to
  training the average-strategy PolicyNet at the end of all iterations.

Both use **algorithm R** reservoir sampling (Vitter 1985). When the buffer is
full, an incoming entry replaces a uniformly-chosen older entry with
probability `capacity / n_seen_total`. This is the property Deep CFR's
convergence proof relies on: the empirical buffer distribution is uniform
over *every* item ever inserted, not just the most-recent capacity-many.
A circular/FIFO buffer would over-weight late iterations and bias the
regret estimates.

Linear CFR weighting (Brown & Sandholm 2019): each entry carries the
iteration index `t` it was inserted at. The training loss multiplies the
per-sample MSE by `t`, so later iterations dominate the gradient.

Storage is plain numpy. We don't shard across processes — the cluster
trainer (cfr_mp_gpu.py) collates traversals from CPU actors over
mp.Queue and the learner appends sequentially.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Sample:
    """Returned by buffer.sample() — torch-free so it crosses mp.Queue cheaply
    and lets the caller decide which device to upload to."""
    x:      np.ndarray   # (B, x_dim)  float32
    target: np.ndarray   # (B, num_actions) float32  (regret_vec or strategy_vec)
    t:      np.ndarray   # (B,)        float32  (CFR iteration index)


class ReservoirBuffer:
    """Algorithm R reservoir. Constant memory, O(1) per insert.

    Invariants:
        * `n_seen` = total inserts attempted (grows forever).
        * `n_filled = min(n_seen, capacity)` = currently-stored entries.
        * Every entry currently in the buffer is a uniform sample from the
          stream of `n_seen` inputs, regardless of insertion order.
    """

    def __init__(self,
                 capacity: int,
                 x_dim: int,
                 target_dim: int,
                 rng: np.random.Generator | None = None):
        self.capacity   = int(capacity)
        self.x_dim      = int(x_dim)
        self.target_dim = int(target_dim)
        self.x      = np.zeros((self.capacity, self.x_dim),      dtype=np.float32)
        self.target = np.zeros((self.capacity, self.target_dim), dtype=np.float32)
        self.t      = np.zeros((self.capacity,),                 dtype=np.float32)
        self.n_seen   = 0
        self.n_filled = 0
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return self.n_filled

    def add(self, x: np.ndarray, target: np.ndarray, t: int) -> None:
        """Insert a single (x, target, t) triple into the reservoir."""
        if self.n_seen < self.capacity:
            i = self.n_seen
            self.x[i]      = x
            self.target[i] = target
            self.t[i]      = float(t)
            self.n_filled += 1
        else:
            # Replace a uniformly-chosen older slot with probability
            # capacity / n_seen. This is the standard algorithm R update;
            # the buffer's contents stay a uniform sample of all inserts.
            j = self.rng.integers(0, self.n_seen + 1)
            if j < self.capacity:
                self.x[j]      = x
                self.target[j] = target
                self.t[j]      = float(t)
        self.n_seen += 1

    def add_batch(self, xs: np.ndarray, targets: np.ndarray, t: int) -> None:
        n = xs.shape[0]
        if n == 0:
            return
        assert xs.shape == (n, self.x_dim), xs.shape
        assert targets.shape == (n, self.target_dim), targets.shape
        # Process one at a time — the algorithm R replacement probability is
        # state-dependent on n_seen, so we can't trivially vectorize without
        # loss of theoretical guarantees. n is typically small (≤ traversal
        # length × n_actors per drain), so the python overhead is fine.
        for k in range(n):
            self.add(xs[k], targets[k], t)

    def sample(self, batch_size: int) -> _Sample:
        assert self.n_filled > 0, "sample from empty buffer"
        idx = self.rng.integers(0, self.n_filled, size=batch_size)
        return _Sample(
            x=self.x[idx].copy(),
            target=self.target[idx].copy(),
            t=self.t[idx].copy(),
        )

    def state_dict(self) -> dict:
        """For checkpointing. Stores only the active region."""
        n = self.n_filled
        return {
            "capacity":   self.capacity,
            "x_dim":      self.x_dim,
            "target_dim": self.target_dim,
            "n_seen":     self.n_seen,
            "n_filled":   self.n_filled,
            "x":          self.x[:n].copy(),
            "target":     self.target[:n].copy(),
            "t":          self.t[:n].copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        assert sd["capacity"]   == self.capacity
        assert sd["x_dim"]      == self.x_dim
        assert sd["target_dim"] == self.target_dim
        n = int(sd["n_filled"])
        self.x[:n]      = sd["x"]
        self.target[:n] = sd["target"]
        self.t[:n]      = sd["t"]
        self.n_seen   = int(sd["n_seen"])
        self.n_filled = n
