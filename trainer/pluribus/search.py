"""Depth-limited online re-solving — Tricks 1 & 2 + nested re-solving.
Pluribus idea → "Stage 2 (online): re-solve the current spot for your whole
range, depth-limited, with continuation strategies at the leaves" · doc §2, §9.

The pipeline (``resolve``):
  1. Bayes the betting line into ranges (ranges.py).
  2. Build the depth-limited public subgame from the agent's LIVE engine state —
     so pot/stacks/exact sizes (incl. an opponent's off-tree size, which is
     already in the state) are exact, no rounding. This reuses the vectorized
     CFR-D machinery of ``rebel_py`` (Subgame/Node/CfrSolver/exploitability) —
     ReBeL and Pluribus share the *search*; what differs is the LEAF:
       • on the river the subgame runs to true terminals (fold/showdown);
       • before the river, a leaf (the street boundary) is valued by the
         CONTINUATION-STRATEGY gadget (Trick 1): an opponent decision node whose
         K children carry the per-hand value of playing the rest of the hand
         under each continuation. CFR then solves the opponent's leaf-choice
         jointly with the subgame → robust to all 4^players continuations.
  3. Solve with vector Linear CFR over all 1326 hands (Trick 2).
  4. Read off the row of the hand actually held (done by the caller, play.py).

NESTED re-solving: an opponent bet outside the agent's size grid is already
reflected in the live engine state (it was applied via the engine, exact), so
re-solving FROM that state prices the exact size with no rounding — call
``resolve`` again. Augmenting an opponent node with a not-yet-taken size is the
deeper variant, noted at ``augment_opponent_size``.
"""
from __future__ import annotations

import numpy as np

from rebel_py.public_tree import (Subgame, Node, make_fold_value,
                                   make_showdown_value)
from rebel_py.subgame_solver import CfrSolver
from rebel_py.exploitability import exploitability_bb
from rebel_py.hand_index import NUM_HANDS, board_free_mask
from rebel_py.pbs import PublicState
from rebel_py import showdown as sd

from .config import PluribusConfig

CHIPS_PER_BB = 100
_FOLD = 0
_STREET_RIVER = 3
_N_VISIBLE = {0: 0, 1: 3, 2: 4, 3: 5, 4: 5}


# ─── small engine-state readers (mirror rebel_py.public_tree) ────────────────

def _total_invested(state) -> tuple[int, int]:
    inv, pri = state.invested_this_street, state.invested_prior_streets
    return int(pri[0] + inv[0]), int(pri[1] + inv[1])


def _fold_player(state) -> int:
    h = state.history
    if len(h) and int(h[-1].type) == _FOLD:
        return int(h[-1].actor)
    return 0


def _public_state_of(env, board) -> PublicState:
    st = env.state()
    to_act = int(env.to_act())
    return PublicState(street=int(st.street), board=list(board), to_act=to_act,
                       pot_bb=int(st.pot_chips) / CHIPS_PER_BB,
                       to_call_bb=int(st.to_call_chips()) / CHIPS_PER_BB,
                       stack_bb=int(st.stacks[to_act]) / CHIPS_PER_BB)


# ─── build a depth-limited subgame rooted at the agent's live env ────────────

def build_engine_subgame(root_env, abstraction, *, depth_limit: int = 1,
                         continuation_fn=None) -> Subgame:
    """Enumerate the abstract betting tree from `root_env`, depth-limited.

    `depth_limit` counts streets below the root before a leaf. A street-closing
    action (or a pre-river all-in runout) produces a LEAF; on the river the tree
    runs to true terminals. Leaves are valued by `continuation_fn(public_state,
    traverser, opp_reach) -> (n_hands,)` (Trick 1) if given, else by a single
    blueprint-free fallback (matched-pot equity is unavailable pre-river without
    a rollout, so a continuation_fn is required for pre-river leaves).
    """
    import pokertrainer_engine as pte

    st0 = root_env.state()
    root_street = int(st0.street)
    board = [int(c) for c in st0.board][:_N_VISIBLE[root_street]]
    # River: precompute the showdown sign matrix once (full 5-card board).
    sign_matrix = None
    if root_street == _STREET_RIVER:
        ranks, valid = sd.hand_ranks(board)
        sign_matrix = sd.showdown_sign_matrix(ranks, valid, board)

    nodes: list[Node] = []

    def add(env, street_depth: int) -> int:
        nid = len(nodes)
        nodes.append(Node(player=-1))
        node = nodes[nid]
        st = env.state()
        if env.is_terminal():
            c0, c1 = _total_invested(st)
            matched = min(c0, c1) / CHIPS_PER_BB
            node.is_terminal = True
            if int(st.terminal) == 1:                      # fold
                node.term_value = make_fold_value(matched, 1 - _fold_player(st), board)
            elif sign_matrix is not None:                  # river showdown
                node.term_value = make_showdown_value(sign_matrix, matched)
            else:                                          # pre-river all-in runout
                node.term_value = _continuation_leaf(env, board, continuation_fn)
            return nid
        cur_street = int(st.street)
        node.player = int(env.to_act())
        ps = _public_state_of(env, board)
        node.public_state = ps
        legal = [int(a) for a in st.legal_actions()]
        abs_legal = abstraction.action.actions(st, legal)
        for a in abs_legal:
            child = env.clone()
            child.step_action(pte.ActionType(a))
            cst = child.state()
            advanced = (not child.is_terminal()) and int(cst.street) > cur_street
            if advanced and street_depth + 1 > depth_limit:
                cid = _leaf_node(nodes, child, board, continuation_fn)
            elif advanced:
                cid = add(child, street_depth + 1)
            else:
                cid = add(child, street_depth)
            node.children.append(cid)
            node.actions.append(a)
        return nid

    add(root_env, 0)
    return Subgame(nodes=nodes, n_hands=NUM_HANDS, board=board,
                   root_pot_bb=int(st0.pot_chips) / CHIPS_PER_BB)


def _leaf_node(nodes, env, board, continuation_fn) -> int:
    nid = len(nodes)
    nb = [int(c) for c in env.state().board][:_N_VISIBLE[int(env.state().street)]]
    nodes.append(Node(player=-1, is_terminal=True,
                      public_state=_public_state_of(env, nb),
                      term_value=_continuation_leaf(env, nb, continuation_fn)))
    return nid


def _continuation_leaf(env, board, continuation_fn):
    """Term-value closure for a depth-limit leaf (Trick 1). The opponent's
    continuation choice is solved by CFR via `continuation_fn`, which returns the
    opponent-best (min for the traverser) per-hand value over the K continuations.
    If no continuation_fn is supplied (e.g. a river-only re-solve never reaches
    here), raise — the caller must provide one for pre-river leaves."""
    if continuation_fn is None:
        raise ValueError("pre-river leaf reached but no continuation_fn provided "
                         "(supply blueprint continuations for depth-limited search)")
    ps = _public_state_of(env, board)

    def fn(traverser: int, opp_reach: np.ndarray) -> np.ndarray:
        return continuation_fn(ps, traverser, opp_reach)
    return fn


# ─── the re-solve ────────────────────────────────────────────────────────────

def resolve(env, blueprint, cfg: PluribusConfig, *, ranges=None,
            continuation_fn=None) -> tuple[np.ndarray, Subgame, list]:
    """Re-solve the subgame at the agent's live `env`. Returns
    (sigma_table, subgame, avg_strategy): `sigma_table[node, hand, action_slot]`
    is implicit in `avg_strategy` (per-node (n_hands, n_children)); the caller
    reads the ROOT node's row for the hand it actually holds.

    `ranges` defaults to the Bayes ranges of `env` under the blueprint."""
    from .ranges import bayes_ranges
    if ranges is None:
        ranges = bayes_ranges(env.state(), blueprint)
    sg = build_engine_subgame(env, blueprint.abstraction,
                              depth_limit=cfg.depth_limit,
                              continuation_fn=continuation_fn)
    solver = CfrSolver(sg, ranges, num_iters=cfg.search_cfr_iters, linear=True)
    solver.multistep()
    avg = solver.average_strategy()
    return _root_action_probs(sg, avg), sg, avg


def _root_action_probs(sg: Subgame, avg: list) -> np.ndarray:
    """(n_hands, NUM_ACTIONS) root strategy, mapped from child-slots to engine
    action ids (0 in slots the abstraction did not offer at the root)."""
    from .config import NUM_ACTIONS
    root = sg.nodes[0]
    out = np.zeros((NUM_HANDS, NUM_ACTIONS), dtype=np.float64)
    if root.is_terminal or avg[0] is None:
        return out
    for k, a in enumerate(root.actions):
        out[:, a] = avg[0][:, k]
    return out


def subgame_exploitability(sg: Subgame, avg: list, ranges) -> float:
    """bb/hand exploitability of `avg` in `sg` (reuses the ReBeL gate)."""
    return exploitability_bb(sg, avg, ranges)
