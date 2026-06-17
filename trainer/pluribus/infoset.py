"""Abstract infoset key + the regret / average-strategy tables.
Pluribus idea → "regret_tables indexed by ABSTRACT infosets" · doc §2, §4.

An infoset (information set) is everything the player to act knows: the street,
whose turn it is, the *bucket* of their private hand (the information
abstraction), and the public betting sequence so far. Two situations that map to
the same key share a strategy — that sharing is exactly what makes the table
small enough to fit in RAM.

    key = (street, to_act, bucket, history_token)

The history token is the chronological tuple of engine ActionType ints applied
so far. Since the MCCFR traversal only ever applies *abstracted* actions, the
engine's recorded history types ARE the abstract betting sequence, so this token
is a faithful, hashable id of the public node. (street/to_act are implied by the
token in HU, but kept explicit for clarity and safety across street closings.)

Tables are plain dicts ``key → np.ndarray[NUM_ACTIONS]`` (float64). They are
SPARSE — an entry is created the first time its infoset is visited. Regret and
strategy-sum are both *additive*, which is what lets the Phase-3 cluster trainer
merge per-worker tables by summation (see blueprint_mp.py).
"""
from __future__ import annotations

import numpy as np

from cfr.regret_matching import regret_matching_np
from .config import NUM_ACTIONS


def infoset_key(state, info_abstraction) -> tuple:
    """Hashable abstract infoset id for the player to act in `state`."""
    bucket = info_abstraction.bucket(state)
    hist = tuple(int(h.type) for h in state.history)
    return (int(state.street), int(state.to_act), bucket, hist)


class Tables:
    """Sparse regret + average-strategy-sum tables for tabular Linear MCCFR."""

    def __init__(self, n_actions: int = NUM_ACTIONS):
        self.n_actions = n_actions
        self.regret: dict[tuple, np.ndarray] = {}
        self.strat: dict[tuple, np.ndarray] = {}

    # ── regret side (the traverser's own infosets) ───────────────────────────
    def regret_vec(self, key: tuple) -> np.ndarray:
        v = self.regret.get(key)
        if v is None:
            v = np.zeros(self.n_actions, dtype=np.float64)
            self.regret[key] = v
        return v

    def add_regret(self, key: tuple, delta: np.ndarray) -> None:
        self.regret_vec(key)[:] += delta

    def strategy(self, key: tuple, mask: np.ndarray) -> np.ndarray:
        """Current strategy σ from regret matching on this infoset's regrets."""
        return regret_matching_np(self.regret_vec(key), mask)

    # ── average-strategy side (accumulated at opponent infosets) ─────────────
    def add_strat(self, key: tuple, delta: np.ndarray) -> None:
        v = self.strat.get(key)
        if v is None:
            v = np.zeros(self.n_actions, dtype=np.float64)
            self.strat[key] = v
        v += delta

    def average_strategy(self, key: tuple, mask: np.ndarray) -> np.ndarray:
        """Blueprint at this infoset: the (linearly weighted) average strategy.

        Falls back to uniform-over-mask for an infoset never accumulated (a hand
        so rare the trainer never sampled it as the opponent's turn)."""
        s = self.strat.get(key)
        if s is None:
            n = max(1.0, float(mask.sum()))
            return (mask / n).astype(np.float64)
        pos = s * mask
        z = pos.sum()
        if z > 0:
            return pos / z
        n = max(1.0, float(mask.sum()))
        return (mask / n).astype(np.float64)

    # ── stats / merge ─────────────────────────────────────────────────────────
    def n_infosets(self) -> int:
        return len(self.regret)

    def merge_from(self, other: "Tables") -> None:
        """Additively fold another Tables (a worker's) into this one."""
        for k, v in other.regret.items():
            cur = self.regret.get(k)
            if cur is None:
                self.regret[k] = v.copy()
            else:
                cur += v
        for k, v in other.strat.items():
            cur = self.strat.get(k)
            if cur is None:
                self.strat[k] = v.copy()
            else:
                cur += v
