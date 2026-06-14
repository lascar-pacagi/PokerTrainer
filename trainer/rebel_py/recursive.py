"""Recursive self-play (the ReBeL outer loop) + the recursive strategy.

Mirrors ``RlRunner`` / ``compute_strategy_recursive`` in the reference
``recursive_solving.cc``.

Self-play (``RlRunner.step``), one game over the (ground-truth) public tree:

  Start at the root with uniform beliefs. Repeatedly:
    1. build a depth-limited subgame rooted at the current node (leaves valued
       by the net);
    2. run a *random* number of CFR iterations (the "play a random CFR iterate"
       safety trick — act_iteration ~ Uniform[0, num_iters]);
    3. sample the next public node from the current-iterate strategy (with
       ``random_action_prob`` exploration for one player) and Bayes-update that
       player's beliefs;
    4. finish the remaining CFR iterations;
    5. emit a training example ``(query@root, root_values_means)`` for *both*
       traversers — this is the net's regression target;
    6. move to the sampled node and recurse until a terminal.

``recursive_strategy`` assembles a full-tree strategy by re-solving a
depth-limited subgame at *every* node and keeping only its root policy (then
propagating beliefs to the children) — the strategy whose exploitability is the
Phase-2 gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .public_tree import Subgame, subtree_subgame
from .subgame_solver import CfrSolver, ValueFn
from .pbs import encode_query, _normalize_safe
from .hand_index import board_free_mask, NUM_HANDS


@dataclass
class SubgameParams:
    depth_limit: int | None = 2   # betting actions before a value-net leaf
    num_iters: int = 64           # CFR iterations per subgame solve
    random_action_prob: float = 0.25
    linear: bool = True
    stop_at_chance: bool = False  # Phase 2b: value the post-chance (river) PBSs


def uniform_beliefs(board) -> list[np.ndarray]:
    free = board_free_mask(board).astype(np.float64) if board else np.ones(NUM_HANDS)
    b = free / free.sum()
    return [b.copy(), b.copy()]


class RlRunner:
    """One self-play actor over a fixed public tree (a fixed board, Phase 2a)."""

    def __init__(self, full: Subgame, value_fn: ValueFn, params: SubgameParams,
                 buffer, rng: np.random.Generator):
        self.full = full
        self.value_fn = value_fn
        self.params = params
        self.buffer = buffer
        self.rng = rng
        self.valid = (board_free_mask(full.board).astype(np.float64)
                      if full.board else np.ones(NUM_HANDS))

    def _solve(self, node_id: int, beliefs):
        sub = subtree_subgame(self.full, node_id, self.params.depth_limit,
                              stop_at_chance=self.params.stop_at_chance)
        return CfrSolver(sub, (beliefs[0], beliefs[1]),
                         num_iters=self.params.num_iters, linear=self.params.linear,
                         value_fn=self.value_fn)

    def _sample_chance(self, node_id: int, beliefs):
        """Nature deals a river card: sample uniformly over the unseen cards, then
        remove the dealt card from both players' ranges (card removal)."""
        node = self.full.nodes[node_id]
        k = int(self.rng.integers(0, len(node.children)))
        r = int(node.chance_cards[k])
        mask = board_free_mask(list(self.full.board) + [r]).astype(np.float64)
        nb = [_normalize_safe(beliefs[0] * mask), _normalize_safe(beliefs[1] * mask)]
        return node.children[k], nb

    def _sample_next(self, node_id: int, solver: CfrSolver, beliefs):
        node = self.full.nodes[node_id]
        p = node.player
        strat = solver.current_strategy()[0]          # (H, n_children) at sub root
        n_act = len(node.children)
        br_sampler = int(self.rng.integers(0, 2))
        if p == br_sampler and self.rng.random() < self.params.random_action_prob:
            a = int(self.rng.integers(0, n_act))
        else:
            bp = beliefs[p]
            s = bp.sum()
            hp = bp / s if s > 0 else self.valid / self.valid.sum()
            hand = int(self.rng.choice(len(hp), p=hp))
            pol = strat[hand].copy()
            tot = pol.sum()
            pol = pol / tot if tot > 0 else np.full(n_act, 1.0 / n_act)
            a = int(self.rng.choice(n_act, p=pol))
        nb = [beliefs[0].copy(), beliefs[1].copy()]
        nb[p] = _normalize_safe(beliefs[p] * strat[:, a])
        return node.children[a], nb

    def _emit(self, node_id: int, solver: CfrSolver, beliefs):
        ps = self.full.nodes[node_id].public_state
        mask = (board_free_mask(ps.board).astype(np.float64)
                if ps is not None and ps.board else self.valid)
        for trav in (0, 1):
            q = encode_query(trav, ps, beliefs[0], beliefs[1]).astype(np.float32)
            tgt = (solver.get_hand_values(trav) * mask).astype(np.float32)
            self.buffer.add(q, tgt)

    def step(self) -> int:
        """Play one self-play game; emit training examples. Returns #examples."""
        node_id = 0
        beliefs = uniform_beliefs(self.full.board)
        emitted = 0
        K = self.params.num_iters
        while not self.full.nodes[node_id].is_terminal:
            if self.full.nodes[node_id].is_chance:    # nature deals the river card
                node_id, beliefs = self._sample_chance(node_id, beliefs)
                continue
            solver = self._solve(node_id, beliefs)
            act_iter = int(self.rng.integers(0, K + 1))
            for i in range(act_iter):
                solver.step(i % 2)
            next_id, next_beliefs = self._sample_next(node_id, solver, beliefs)
            for i in range(act_iter, K):
                solver.step(i % 2)
            self._emit(node_id, solver, beliefs)
            emitted += 2
            node_id, beliefs = next_id, next_beliefs
        return emitted


def recursive_strategy(full: Subgame, value_fn: ValueFn, params: SubgameParams):
    """Full-tree strategy by *consistent* depth-limited resolving (the gate).

    Mirrors ``compute_strategy_recursive_to_leaf``: at a node, solve its
    depth-limited subgame ONCE and read the strategy for *every* node in that
    partial tree off that single solve (so the interior nodes stay mutually
    consistent — one equilibrium). Only at the partial tree's non-terminal
    leaves (the value-net leaves) do we start a fresh subgame, with the beliefs
    propagated (unnormalized) down to that leaf and renormalized there.

    Re-solving each node independently instead (the naive version) is unsafe and
    leaves the assembled strategy exploitable even with a perfect value net.
    """
    n = len(full.nodes)
    strat: list = [None] * n
    for nid, nd in enumerate(full.nodes):
        if not nd.is_terminal and not nd.is_net_leaf and not nd.is_chance:
            a = len(nd.children)
            strat[nid] = np.full((NUM_HANDS, a), 1.0 / a)

    def rec(root_full_id: int, beliefs):
        sub = subtree_subgame(full, root_full_id, params.depth_limit,
                              stop_at_chance=params.stop_at_chance)
        sv = CfrSolver(sub, (beliefs[0], beliefs[1]), num_iters=params.num_iters,
                       linear=params.linear, value_fn=value_fn)
        sv.multistep()
        avg = sv.average_strategy()
        # BFS the partial tree and the full tree in lockstep (subtree_subgame
        # preserves child order). Beliefs propagate UNNORMALIZED within the
        # partial tree; renormalize only when opening a fresh subgame at a leaf.
        queue = [(0, root_full_id, beliefs)]
        while queue:
            sub_id, full_id, bel = queue.pop()
            snd = sub.nodes[sub_id]
            if snd.is_terminal:
                continue
            if snd.is_net_leaf:                      # partial-tree leaf → re-solve
                rec(full_id, [_normalize_safe(bel[0]), _normalize_safe(bel[1])])
                continue
            fnode = full.nodes[full_id]
            if snd.is_chance:                        # nature: no strategy, mask beliefs
                for k, (sc, fc) in enumerate(zip(snd.children, fnode.children)):
                    mask = board_free_mask(list(sub.board) + [int(snd.chance_cards[k])])
                    queue.append((sc, fc, [bel[0] * mask, bel[1] * mask]))
                continue
            strat[full_id] = avg[sub_id]             # consistent: from THIS solve
            p = snd.player
            for k, (sc, fc) in enumerate(zip(snd.children, fnode.children)):
                nb = [bel[0].copy(), bel[1].copy()]
                nb[p] = bel[p] * avg[sub_id][:, k]
                queue.append((sc, fc, nb))

    rec(0, uniform_beliefs(full.board))
    return strat
