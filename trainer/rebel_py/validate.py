"""Phase-1 ground-truth validation of the CFR-D solver (no value net).

(a) Several river subgames: solve, assert exploitability → ~0 (the rigorous,
    exact correctness proof on the full 1326-combo space).
(b) 10bb push/fold: solve the preflop jam/fold game at the 169 strategic-class
    level (card removal via the joint-count matrix C, equity via E), and check
    the recovered SB-jam / BB-call ranges agree with the existing
    ``evaluate.pushfold_solver`` Nash oracle.

Run:
    PYTHONPATH=engine/build:trainer python -m rebel_py.validate
"""
from __future__ import annotations

import numpy as np

from .public_tree import Subgame, Node, build_river_subgame, make_fold_value
from .subgame_solver import CfrSolver
from .exploitability import exploitability_bb
from .hand_index import board_free_mask


# ─── (a) river ──────────────────────────────────────────────────────────────

def _uniform_beliefs(sg: Subgame):
    free = board_free_mask(sg.board).astype(np.float64) if sg.board \
        else np.ones(sg.n_hands)
    b = free / free.sum()
    return (b.copy(), b.copy())


def validate_river(seeds=(7, 11, 23, 42), iters=1000) -> None:
    # All-terminal action sets only (no value net in Phase 1): fold/call/jam,
    # plus a single intermediate size. A full 11-action river at 100bb explodes
    # into unbounded re-raise wars — that needs the depth-limit value net (P2).
    print("── river endgames (1326 combos, exact showdown) ──")
    for seed in seeds:
        for stack in (20, 100):
            sg = build_river_subgame(seed=seed, starting_stack_bb=stack,
                                     allowed_actions=(0, 1, 10))
            beliefs = _uniform_beliefs(sg)
            sv = CfrSolver(sg, beliefs, num_iters=iters, linear=True)
            sv.multistep()
            e = exploitability_bb(sg, sv.average_strategy(), beliefs)
            print(f"  seed {seed:2d} stack {stack:3d}bb [F/C/AI] nodes={len(sg.nodes):3d}  "
                  f"exploitability={e:.5f} bb/hand")
            assert e < 0.05, f"river exploitability too high: {e}"
    print("  OK: all river subgames converged to ~0 exploitability ✓\n")


# ─── (b) push/fold vs the Nash oracle ───────────────────────────────────────

def _pushfold_subgame(E, C, combos, stack_bb, sb_blind=0.5, bb_blind=1.0):
    """169-class preflop jam/fold tree.

    Card-removal-correct terminal CVs: removal matrix Rm[i][j] = coexist
    fraction = C[i][j]/(combos_i*combos_j); showdown payoff Pm[i][j] = 2E-1.
    """
    Rm = C.astype(np.float64) / (combos[:, None] * combos[None, :])
    Pm = 2.0 * E - 1.0
    callM = stack_bb * (Pm * Rm)

    def fold_term(matched, winner):
        def fn(traverser, opp_reach):
            sign = 1.0 if traverser == winner else -1.0
            return sign * matched * (Rm @ opp_reach)
        return fn

    def call_term():
        def fn(traverser, opp_reach):
            return callM @ opp_reach
        return fn

    nodes = [
        Node(player=0, children=[1, 2], actions=[0, 10]),       # 0: SB fold/jam
        Node(player=-1, is_terminal=True,                       # 1: SB folds
             term_value=fold_term(sb_blind, winner=1)),
        Node(player=1, children=[3, 4], actions=[0, 1]),        # 2: BB fold/call
        Node(player=-1, is_terminal=True,                       # 3: BB folds
             term_value=fold_term(bb_blind, winner=0)),
        Node(player=-1, is_terminal=True, term_value=call_term()),  # 4: showdown
    ]
    return Subgame(nodes=nodes, n_hands=len(combos), board=[], root_pot_bb=1.5)


def validate_pushfold(stack_bb=10, n_deals=4_000_000, iters=4000) -> None:
    print("── 10bb push/fold vs the Nash oracle (169 classes) ──")
    import pokertrainer_engine as pte
    from evaluate.preflop_equity import build_equity_matrix, COMBOS, CLASS_LABELS
    from evaluate import pushfold_solver as ps

    ev = pte.HandEvaluator.load_or_generate("")
    E, C = build_equity_matrix(ev, n_deals, seed=7)
    combos = COMBOS.astype(np.float64)

    sg = _pushfold_subgame(E, C, combos, stack_bb)
    beliefs = (combos / combos.sum(), combos / combos.sum())
    sv = CfrSolver(sg, beliefs, num_iters=iters, linear=True)
    sv.multistep()
    avg = sv.average_strategy()
    # SB-jam freq per class = avg strategy at root (node 0), action slot 1 (jam).
    sb_jam = avg[0][:, 1]
    # BB-call freq per class = avg at node 2, slot 1 (call).
    bb_call = avg[2][:, 1]

    oracle = ps.solve_pushfold(E, C, stack_bb, iters=3000)
    w = combos
    def agree(a, b):
        return float((w * ((a >= 0.5) == (b >= 0.5))).sum() / w.sum())
    sj = agree(sb_jam, oracle.sb_jam)
    bc = agree(bb_call, oracle.bb_call)
    sj_pct = float((w * sb_jam).sum() / w.sum())
    e = exploitability_bb(sg, avg, beliefs)
    print(f"  exploitability={e*1000:.1f} mbb/hand")
    print(f"  SB-jam : ReBeL {sj_pct*100:4.1f}%  vs oracle {oracle.sb_jam_pct*100:4.1f}%  "
          f"agree={sj*100:.1f}%")
    print(f"  BB-call: agree={bc*100:.1f}%")
    assert sj > 0.9 and bc > 0.9, "push/fold ranges disagree with the oracle"
    print("  OK: ReBeL CFR-D reproduces the push/fold Nash oracle ✓\n")


# ─── (c) turn→river chance node (Phase 2b ground truth) ─────────────────────

def validate_turn(seeds=(7, 11), stack_bb=6, iters=400) -> None:
    """Full ground-truth turn→river tree (a real chance node: the river card)
    must solve to ~0 exploitability — the exact check on the chance-node
    card-removal math (per-card showdown matrices + 1/44 matchup normalizer)."""
    from .public_tree import build_turn_subgame
    print("── turn→river endgames (chance node = river card) ──")
    for seed in seeds:
        sg = build_turn_subgame(seed=seed, starting_stack_bb=stack_bb,
                                allowed_actions=(0, 1, 10))
        beliefs = _uniform_beliefs(sg)
        sv = CfrSolver(sg, beliefs, num_iters=iters, linear=True)
        sv.multistep()
        e = exploitability_bb(sg, sv.average_strategy(), beliefs)
        nch = sum(n.is_chance for n in sg.nodes)
        print(f"  seed {seed:2d} stack {stack_bb}bb  nodes={len(sg.nodes):4d} "
              f"chance={nch}  exploitability={e:.5f} bb/hand")
        assert e < 0.05, f"turn→river exploitability too high: {e}"
    print("  OK: turn→river ground truth converged to ~0 exploitability ✓\n")


def main():
    validate_river()
    validate_pushfold()
    validate_turn()
    print("Validation PASSED.")


if __name__ == "__main__":
    main()
