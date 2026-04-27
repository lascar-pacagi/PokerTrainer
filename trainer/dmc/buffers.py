"""Circular replay buffer for DMC transitions.

Flat-sequence encoding (v0.2): each transition stores:
    x  — state vector (X_DIM,)         float32   # includes flat action history
    a  — taken action feature (A_DIM,) float32
    r  — terminal Monte-Carlo return, in BB (scalar float32)

DouZero uses shared-memory (T, B) buffers indexed by multi-process actors; for
Phase 1.1 we start with a single-process in-memory buffer sized by total
transitions. The interface is append(one transition) and sample(batch_size),
which is enough to drive gradient steps. When we scale to multiprocess we swap
this out for torch.multiprocessing shared tensors without changing the rest of
the code.
"""
from __future__ import annotations

import numpy as np


class MCBuffer:
    # Layout offsets inside x/a so slot_distribution() can decode street and
    # taken-slot directly from the buffer without re-encoding. See
    # docs/STATE_ENCODING.md and engine/src/encoder.h for the source of truth.
    STREET_OFFSET = 104   # x[104:108] is one-hot street (preflop/flop/turn/river)
    N_STREETS     = 4
    SLOT_OFFSET   = 0     # a[0:11] is one-hot action slot
    N_SLOTS       = 11

    def __init__(self, capacity: int, x_dim: int, a_dim: int,
                 rng: np.random.Generator | None = None,
                 reward_clip: float | None = None):
        self.capacity = int(capacity)
        self.x = np.zeros((capacity, x_dim), dtype=np.float32)
        self.a = np.zeros((capacity, a_dim), dtype=np.float32)
        self.r = np.zeros((capacity,),       dtype=np.float32)
        self.n = 0       # items written (grows to capacity then stops tracking)
        self.head = 0    # next write position
        self.rng = rng or np.random.default_rng()
        # Symmetric clip on terminal MC return (in BB). Applied at SAMPLE
        # time, not append time, so the trainer can schedule the clip across
        # training (tight early, looser later) without the buffer holding
        # stale clipped values. Set to None to disable.
        self.reward_clip = float(reward_clip) if reward_clip is not None else None

    def __len__(self) -> int:
        return min(self.n, self.capacity)

    def set_reward_clip(self, value: float | None) -> None:
        """Update the sample-time clip. Safe to call between grad steps."""
        self.reward_clip = float(value) if value is not None else None

    def append(self, x: np.ndarray, a: np.ndarray, r: float) -> None:
        i = self.head % self.capacity
        self.x[i] = x
        self.a[i] = a
        self.r[i] = float(r)  # store raw — clip is applied in sample()
        self.head += 1
        self.n = max(self.n, self.head)

    def append_trajectory(self, xs, as_, terminal_return: float) -> None:
        """Append one trajectory's (state, action) pairs with the same terminal
        return as each transition's target (pure MC).
        """
        for x, a in zip(xs, as_):
            self.append(x, a, terminal_return)

    def sample(self, batch_size: int) -> dict:
        assert len(self) > 0, "sample from empty buffer"
        idx = self.rng.integers(0, len(self), size=batch_size)
        r = self.r[idx].copy()
        if self.reward_clip is not None:
            np.clip(r, -self.reward_clip, self.reward_clip, out=r)
        return {
            "x": self.x[idx].copy(),
            "a": self.a[idx].copy(),
            "r": r,
        }

    def slot_distribution(self) -> np.ndarray:
        """Per-street per-action-slot counts over the active buffer region.

        Returns: int64 array (N_STREETS, N_SLOTS). Reads the one-hot street
        from x[..., 104:108] and the one-hot action slot from a[..., 0:11].
        Used as a training-side diagnostic — if the rollout distribution has
        already collapsed (e.g. AI=1.0 preflop, postflop n=0), the eval-time
        slot summary will too, but we'll see it sooner here.
        """
        n = len(self)
        counts = np.zeros((self.N_STREETS, self.N_SLOTS), dtype=np.int64)
        if n == 0:
            return counts
        streets = self.x[:n, self.STREET_OFFSET:self.STREET_OFFSET + self.N_STREETS].argmax(axis=1)
        slots = self.a[:n, self.SLOT_OFFSET:self.SLOT_OFFSET + self.N_SLOTS].argmax(axis=1)
        np.add.at(counts, (streets, slots), 1)
        return counts

    @staticmethod
    def format_slot_distribution(counts: np.ndarray) -> str:
        """Compact one-line-per-street summary of slot_distribution() output.

        Format mirrors evaluate.match.MatchResult.slot_summary so the eyes
        learn one shape: F C R25 R33 R50 R75 R100 R150 R200 R300 AI.
        """
        labels = ["F", "C", "R25", "R33", "R50", "R75", "R100", "R150", "R200", "R300", "AI"]
        streets = ["pre ", "flop", "turn", "rvr "]
        lines = []
        for s in range(MCBuffer.N_STREETS):
            n = int(counts[s].sum())
            if n == 0:
                lines.append(f"  [buf-{streets[s]}] n=0")
                continue
            freqs = counts[s] / n
            cells = "  ".join(f"{lab}={freq:.2f}" for lab, freq in zip(labels, freqs))
            lines.append(f"  [buf-{streets[s]}] n={n:<6} {cells}")
        return "\n".join(lines)
