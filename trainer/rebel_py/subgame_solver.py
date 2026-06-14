"""Vectorized CFR-D over a public subgame tree (Phase 1: no value net).

Faithful port of the reference `CFR` in
``trainer/rebel/rebel/csrc/liars_dice/subgame_solving.cc`` (alternating
traverser, cumulative-regret matching, reach-weighted average strategy,
Linear-CFR / DCFR discounts, ``root_values_means`` as the value-net target),
but vectorized over the 1326 hands: every per-node quantity is an
``(n_hands, n_children)`` or ``(n_hands,)`` array.

Per node we keep (decision nodes only): cumulative regret, the current strategy
(regret matching), and the reach-weighted cumulative strategy (→ average).
Per CFR step for a traverser we: compute both players' reaches under the current
strategy, set leaf values from the terminal closures (opponent-reach weighted),
back values up the tree accumulating regret at traverser nodes, then regret-match
and accumulate the average. `get_hand_values(p)` returns the running-mean root
counterfactual values — the training target for the value net.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .public_tree import Subgame
from .pbs import encode_query, query_dim
from .hand_index import board_free_mask

_EPS = 1e-80

# A value function maps a (N, query_dim) batch of PBS queries to (N, NUM_HANDS)
# per-hand values for the traverser, assuming normalized opponent beliefs.
ValueFn = Callable[[np.ndarray], np.ndarray]


class CfrSolver:
    def __init__(self, subgame: Subgame, beliefs: tuple[np.ndarray, np.ndarray],
                 *, num_iters: int = 1000, linear: bool = True,
                 value_fn: Optional[ValueFn] = None):
        self.sg = subgame
        self.nodes = subgame.nodes
        self.H = subgame.n_hands
        self.beliefs = (np.asarray(beliefs[0], dtype=np.float64),
                        np.asarray(beliefs[1], dtype=np.float64))
        self.num_iters = num_iters
        self.linear = linear
        self.value_fn = value_fn

        # Depth-limit value-net leaves: queried each step for the current
        # traverser, scaled by the opponent's reach mass at the leaf.
        self.net_leaves = [nid for nid, nd in enumerate(self.nodes)
                           if nd.is_net_leaf]
        if self.net_leaves and value_fn is None:
            raise ValueError("subgame has value-net leaves but no value_fn given")
        self._qdim = query_dim()
        self._valid = (board_free_mask(subgame.board).astype(np.float64)
                       if subgame.board else np.ones(self.H))

        # Chance nodes (Phase 2b): one dealt board card per child. Precompute the
        # per-child card-removal mask and the matchup normalizer. At the turn the
        # river is 1 of (52 - 4 board - 2 hero - 2 opp) = 44 equally-likely cards.
        self.chance_nodes = [nid for nid, nd in enumerate(self.nodes) if nd.is_chance]
        self.chance_masks: dict[int, np.ndarray] = {}
        self.chance_div = float(52 - len(subgame.board) - 4) if subgame.board else 1.0
        for nid in self.chance_nodes:
            nd = self.nodes[nid]
            masks = np.stack([board_free_mask(list(subgame.board) + [int(r)])
                              for r in nd.chance_cards]).astype(np.float64)
            self.chance_masks[nid] = masks  # (n_children, H)

        # parent[node], slot[node] = index of `node` within parent.children.
        n = len(self.nodes)
        self.parent = np.full(n, -1, dtype=np.int64)
        self.slot = np.full(n, -1, dtype=np.int64)
        for nid, node in enumerate(self.nodes):
            for k, c in enumerate(node.children):
                self.parent[c] = nid
                self.slot[c] = k

        # Per-decision-node arrays (None for terminals).
        self.regret: list[np.ndarray | None] = [None] * n
        self.cur: list[np.ndarray | None] = [None] * n        # current strategy
        self.sumstrat: list[np.ndarray | None] = [None] * n   # reach-weighted cum
        self.avg: list[np.ndarray | None] = [None] * n        # average strategy
        for nid, node in enumerate(self.nodes):
            if not node.is_terminal and not node.is_net_leaf and not node.is_chance:
                a = len(node.children)
                self.regret[nid] = np.zeros((self.H, a))
                self.cur[nid] = np.full((self.H, a), 1.0 / a)
                self.sumstrat[nid] = np.zeros((self.H, a))
                self.avg[nid] = np.full((self.H, a), 1.0 / a)

        self.reach = [np.zeros((n, self.H)), np.zeros((n, self.H))]
        self.values = np.zeros((n, self.H))
        self.root_mean = [np.zeros(self.H), np.zeros(self.H)]
        self.steps = [0, 0]

    # ─── reach (top-down) under a given per-node strategy ────────────────────
    def _compute_reach(self, player: int, strat: list, out: np.ndarray) -> None:
        out[0] = self.beliefs[player]
        for nid in range(1, len(self.nodes)):
            par = int(self.parent[nid])
            par_node = self.nodes[par]
            if par_node.is_chance:
                out[nid] = out[par] * self.chance_masks[par][int(self.slot[nid])]
            elif par_node.player == player:
                out[nid] = out[par] * strat[par][:, int(self.slot[nid])]
            else:
                out[nid] = out[par]

    # ─── value-net leaves (batched query, opp-reach-mass scaled) ─────────────
    def _compute_net_leaf_values(self, traverser: int) -> None:
        opp = 1 - traverser
        N = len(self.net_leaves)
        queries = np.empty((N, self._qdim))
        for row, nid in enumerate(self.net_leaves):
            ps = self.nodes[nid].public_state
            queries[row] = encode_query(traverser, ps,
                                        self.reach[0][nid], self.reach[1][nid])
        vals = np.asarray(self.value_fn(queries), dtype=np.float64)  # (N, H), normalized
        for row, nid in enumerate(self.net_leaves):
            mass = float(self.reach[opp][nid].sum())
            self.values[nid] = vals[row] * self._valid * mass

    # ─── one alternating CFR step for `traverser` ────────────────────────────
    def step(self, traverser: int) -> None:
        opp = 1 - traverser
        self._compute_reach(0, self.cur, self.reach[0])
        self._compute_reach(1, self.cur, self.reach[1])
        if self.net_leaves:
            self._compute_net_leaf_values(traverser)

        # leaf values (opponent-reach weighted), then back up.
        for nid in range(len(self.nodes) - 1, -1, -1):
            node = self.nodes[nid]
            if node.is_terminal:
                self.values[nid] = node.term_value(traverser, self.reach[opp][nid])
                continue
            if node.is_net_leaf:
                continue  # value already set by _compute_net_leaf_values
            if node.is_chance:
                v = np.zeros(self.H)
                for c in node.children:
                    v += self.values[c]
                self.values[nid] = v / self.chance_div
                continue
            if node.player == traverser:
                v = np.zeros(self.H)
                reg = self.regret[nid]
                cur = self.cur[nid]
                for k, c in enumerate(node.children):
                    av = self.values[c]
                    reg[:, k] += av
                    v += av * cur[:, k]
                reg -= v[:, None]
                self.values[nid] = v
            else:
                v = np.zeros(self.H)
                for c in node.children:
                    v += self.values[c]
                self.values[nid] = v

        # root running-mean value (the training target).
        rv = self.values[0]
        alpha = 2.0 / (self.steps[traverser] + 2) if self.linear \
            else 1.0 / (self.steps[traverser] + 1)
        self.root_mean[traverser] += (rv - self.root_mean[traverser]) * alpha

        num = self.steps[traverser] + 1
        disc = num / (num + 1.0) if self.linear else 1.0

        # regret matching at the traverser's nodes → new current strategy.
        for nid, node in enumerate(self.nodes):
            if node.is_terminal or node.is_net_leaf or node.player != traverser:
                continue
            pos = np.maximum(self.regret[nid], _EPS)
            self.cur[nid] = pos / pos.sum(axis=1, keepdims=True)

        # reach (traverser) under the NEW strategy → reach-weight the average.
        self._compute_reach(traverser, self.cur, self.reach[traverser])
        for nid, node in enumerate(self.nodes):
            if node.is_terminal or node.is_net_leaf or node.player != traverser:
                continue
            if self.linear:
                self.regret[nid] *= disc
                self.sumstrat[nid] *= disc
            self.sumstrat[nid] += self.reach[traverser][nid][:, None] * self.cur[nid]
            s = self.sumstrat[nid].sum(axis=1, keepdims=True)
            a = self.cur[nid].shape[1]
            safe = np.where(s > 0, s, 1.0)
            self.avg[nid] = np.where(s > 0, self.sumstrat[nid] / safe, 1.0 / a)

        self.steps[traverser] += 1

    def multistep(self) -> None:
        for it in range(self.num_iters):
            self.step(it % 2)

    def get_hand_values(self, player: int) -> np.ndarray:
        return self.root_mean[player]

    def average_strategy(self) -> list:
        return self.avg

    def current_strategy(self) -> list:
        """The last (current-iterate) strategy — used to sample the next public
        state and to propagate beliefs during self-play (reference:
        `get_sampling_strategy` / `get_belief_propogation_strategy`)."""
        return self.cur
