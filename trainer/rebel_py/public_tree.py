"""Public game tree for a depth-limited subgame + terminal value functions.

A node is a *public* state (betting sequence) — it does not depend on the hole
cards, since betting legality/pot/stacks are hole-independent. Each decision
node has one child per legal action (the 11-action abstraction). Leaves are
either true terminals (fold / showdown), or — at the depth limit (next chance
node) — value-net leaves (Phase 2+). Phase 1 builds trees whose leaves are all
true terminals.

Terminal per-hand counterfactual values (vectorized over the 1326 hands) are
provided as closures stored on the node:

    term_value(traverser: int, opp_reach: np.ndarray) -> np.ndarray   # (n_hands,)

Showdown uses a board-specific sign matrix (showdown.py); fold uses the matched
pot. Both are masked for card removal.

The river builder reuses the engine: deal a hand, limp/check to the river (so
the engine deals a real board), then enumerate the river betting tree by cloning
+ stepping `Env`. River is the last street, so there are no chance nodes — every
leaf is a true terminal — which makes it the clean Phase-1 solver validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import showdown as sd
from .hand_index import NUM_HANDS, CONFLICT, board_free_mask
from .pbs import PublicState

CHIPS_PER_BB = 100
_FOLD, _CHECK_CALL, _ALL_IN = 0, 1, 10


@dataclass
class Node:
    player: int                      # 0=SB / 1=BB acting; -1 if terminal/leaf
    children: list[int] = field(default_factory=list)   # child node indices
    actions: list[int] = field(default_factory=list)     # engine ActionType per child
    is_terminal: bool = False
    term_value: Optional[Callable[[int, np.ndarray], np.ndarray]] = None
    # Depth-limit value-net leaf: a non-terminal leaf whose continuation value
    # comes from the value net (queried via `public_state`). Phase 2+.
    is_net_leaf: bool = False
    public_state: Optional[PublicState] = None


@dataclass
class Subgame:
    nodes: list[Node]
    n_hands: int
    board: list[int]                 # 5 cards (river) or [] (preflop)
    root_pot_bb: float


# ─── terminal value closures ────────────────────────────────────────────────

def make_fold_value(matched_bb: float, winner: int, board: list[int]):
    """Hand-independent: +/-matched times opponent reach mass (card removal)."""
    free = board_free_mask(board) if board else np.ones(NUM_HANDS)
    compat = (~CONFLICT).astype(np.float64) * free[None, :]

    def fn(traverser: int, opp_reach: np.ndarray) -> np.ndarray:
        sign = 1.0 if traverser == winner else -1.0
        return sign * matched_bb * (compat @ opp_reach)
    return fn


def make_showdown_value(sign_matrix: np.ndarray, stake_bb: float):
    """v[h] = stake * sum_j sign(h beats j) * opp_reach[j]. Same for both
    traversers (sign is 'row hand beats column hand')."""
    def fn(traverser: int, opp_reach: np.ndarray) -> np.ndarray:
        return stake_bb * (sign_matrix @ opp_reach)
    return fn


# ─── engine-driven river subgame ────────────────────────────────────────────

def _total_invested(state):
    inv = state.invested_this_street
    pri = state.invested_prior_streets
    return [int(pri[0] + inv[0]), int(pri[1] + inv[1])]


def _public_state_of(env, board: list[int]) -> PublicState:
    """Read the hole-independent public scalars at a live decision node."""
    st = env.state()
    to_act = int(env.to_act())
    return PublicState(
        street=int(st.street),
        board=list(board),
        to_act=to_act,
        pot_bb=int(st.pot_chips) / CHIPS_PER_BB,
        to_call_bb=int(st.to_call_chips()) / CHIPS_PER_BB,
        stack_bb=int(st.stacks[to_act]) / CHIPS_PER_BB,
    )


def build_river_subgame(seed: int = 7,
                        starting_stack_bb: int = 100,
                        allowed_actions: tuple[int, ...] | None = None,
                        max_betting_depth: int | None = None) -> Subgame:
    """Reach a river state by limping/checking, then enumerate its betting tree.

    `allowed_actions` restricts the abstraction (e.g. (FOLD, CHECK_CALL, ALL_IN)
    for a small tree); None = all engine-legal actions.

    `max_betting_depth` (Phase 2+) truncates the betting tree at the given depth
    (counted in actions from the subgame root): a non-terminal node at the limit
    becomes a value-net leaf carrying its `PublicState`, instead of recursing.
    None = enumerate to true terminals (Phase 1).
    """
    import pokertrainer_engine as pte

    root_env = pte.Env(seed, starting_stack_bb * CHIPS_PER_BB)
    root_env.reset(seed)
    # Limp/check to the river: apply CHECK_CALL until street == RIVER (3) or terminal.
    safety = 16
    while not root_env.is_terminal() and safety > 0:
        st = root_env.state()
        if int(st.street) >= 3:  # RIVER
            break
        legal = [int(a) for a in root_env.observation().legal]
        # CHECK_CALL is always legal.
        idx = legal.index(_CHECK_CALL)
        root_env.step(idx)
        safety -= 1
    st = root_env.state()
    assert int(st.street) == 3 and not root_env.is_terminal(), \
        f"failed to reach a live river (street={int(st.street)}, term={root_env.is_terminal()})"

    board = [int(c) for c in st.board]
    # Precompute the river showdown sign matrix once (shared by all showdowns).
    ranks, valid = sd.hand_ranks(board)
    sign_matrix = sd.showdown_sign_matrix(ranks, valid, board)

    nodes: list[Node] = []

    def add(env, depth) -> int:
        nid = len(nodes)
        nodes.append(Node(player=-1))  # placeholder; filled below
        node = nodes[nid]
        if env.is_terminal():
            node.is_terminal = True
            stt = env.state()
            c0, c1 = _total_invested(stt)
            matched_bb = min(c0, c1) / CHIPS_PER_BB
            reason = int(stt.terminal)  # 1=FOLD, 2=SHOWDOWN
            if reason == 1:  # fold: whoever played FOLD loses → other wins
                winner = 1 - _fold_player(stt)
                node.term_value = make_fold_value(matched_bb, winner, board)
            else:  # showdown
                node.term_value = make_showdown_value(sign_matrix, matched_bb)
            return nid
        # decision node — record its public state (cheap; enables re-rooting).
        node.player = int(env.to_act())
        node.public_state = _public_state_of(env, board)
        if max_betting_depth is not None and depth >= max_betting_depth:
            node.is_net_leaf = True
            return nid
        legal = [int(a) for a in env.observation().legal]
        if allowed_actions is not None:
            legal = [a for a in legal if a in allowed_actions]
        for a in legal:
            child = env.clone()
            child.step_action(pte.ActionType(a))
            cid = add(child, depth + 1)
            node.children.append(cid)
            node.actions.append(a)
        return nid

    add(root_env, 0)
    return Subgame(nodes=nodes, n_hands=NUM_HANDS, board=board,
                   root_pot_bb=int(st.pot_chips) / CHIPS_PER_BB)


def _fold_player(state) -> int:
    """Which player folded, read from the last history action (type FOLD=0)."""
    hist = state.history
    if len(hist) > 0 and int(hist[-1].type) == _FOLD:
        return int(hist[-1].actor)
    # Fallback: shouldn't happen for a fold terminal.
    return 0


# ─── re-rooting / depth-limiting an already-built tree ───────────────────────

def subtree_subgame(full: Subgame, root_nid: int,
                    max_depth: int | None) -> Subgame:
    """A depth-limited subgame rooted at `root_nid` of an existing tree.

    Copies the subtree from `root_nid`, renumbering nodes from 0. Decision nodes
    at `max_depth` (counted from the new root) become value-net leaves, carrying
    the `public_state` recorded at build time. True terminals and pre-existing
    net leaves are copied verbatim. This is how the ReBeL recursion re-solves a
    fresh subgame at each sampled public state without touching the engine.
    """
    old = full.nodes
    new_nodes: list[Node] = []

    def copy(old_id: int, depth: int) -> int:
        on = old[old_id]
        nn = Node(player=on.player, is_terminal=on.is_terminal,
                  term_value=on.term_value, is_net_leaf=on.is_net_leaf,
                  public_state=on.public_state)
        new_id = len(new_nodes)
        new_nodes.append(nn)
        if on.is_terminal or on.is_net_leaf:
            return new_id
        if max_depth is not None and depth >= max_depth:
            nn.is_net_leaf = True   # truncate this decision node into a leaf
            return new_id
        for a, c in zip(on.actions, on.children):
            cid = copy(c, depth + 1)
            nn.children.append(cid)
            nn.actions.append(a)
        return new_id

    copy(root_nid, 0)
    ps = old[root_nid].public_state
    root_pot = ps.pot_bb if ps is not None else full.root_pot_bb
    return Subgame(nodes=new_nodes, n_hands=full.n_hands, board=full.board,
                   root_pot_bb=root_pot)
