"""The Pluribus agent: blueprint where it is dense, search where it pays.
Pluribus idea → "The full play-time loop": preflop play the blueprint directly;
elsewhere Bayes the ranges and re-solve, then look at your cards · doc §2, §9.

    act(history, board, my_hand):
        if preflop and on-tree:  return sample(blueprint[abstract(...)])
        ranges = bayes_ranges(history, board, assumed=blueprint)
        sigma  = resolve(public_state, ranges)         # whole-range, depth-limited
        return sample(sigma[my_hand])                  # MY cards, only at the end

This wires the two stages. The river re-solve is exact (true terminals, no
continuations); flop/turn re-solving needs a postflop-trained blueprint to value
the continuation leaves — without one the agent plays the blueprint there (and
says so), so a Phase-1-only (preflop) blueprint still yields a runnable agent
whose river decisions are search-improved.
"""
from __future__ import annotations

import numpy as np

from rebel_py.hand_index import hand_index
from .config import PluribusConfig
from .search import resolve

_PREFLOP, _RIVER = 0, 3


class PluribusAgent:
    """Plays a hand from a live engine Env, one decision at a time."""

    def __init__(self, blueprint, cfg: PluribusConfig | None = None,
                 rng: np.random.Generator | None = None,
                 search_streets: tuple[int, ...] = (_RIVER,)):
        self.bp = blueprint
        self.cfg = cfg or PluribusConfig()
        self.rng = rng or np.random.default_rng(0)
        # Streets on which to RE-SOLVE (vs play the blueprint). River by default
        # (exact, no continuation rollout needed); add 1/2 once a postflop
        # blueprint + continuation values are available.
        self.search_streets = set(search_streets)

    def act(self, env) -> int:
        """Return the engine ActionType int to play at the agent's turn."""
        st = env.state()
        street = int(st.street)
        if street not in self.search_streets:
            # Blueprint stage (dense preflop abstraction; postflop fallback).
            return self.bp.sample(st, self.rng)

        # Search stage: re-solve the whole range, then read our row.
        probs, _, _ = resolve(env, self.bp, self.cfg)
        hero = int(env.to_act())
        hole = st.hole[hero]
        h = hand_index(int(hole[0]), int(hole[1]))
        row = probs[h]
        z = row.sum()
        if z <= 0:                              # search put no mass (shouldn't happen)
            return self.bp.sample(st, self.rng)
        idx = np.nonzero(row)[0]
        return int(self.rng.choice(idx, p=row[idx] / row[idx].sum()))
