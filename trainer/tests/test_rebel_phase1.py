"""Phase-1 ReBeL tests: hand indexing, showdown values, CFR-D correctness.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_rebel_phase1.py
"""
from __future__ import annotations

import rebel_py  # noqa: F401  (pins BLAS threads before numpy loads)
import numpy as np

from rebel_py.hand_index import (NUM_HANDS, HAND_CARDS, hand_index, CONFLICT,
                                 board_free_mask, card_str)
from rebel_py import showdown as sd
from rebel_py.public_tree import build_river_subgame
from rebel_py.subgame_solver import CfrSolver
from rebel_py.exploitability import exploitability_bb


def test_hand_index():
    assert NUM_HANDS == 1326
    assert HAND_CARDS.shape == (1326, 2)
    # round-trip a few
    for a, b in [(0, 1), (12 * 4 + 2, 12 * 4 + 3), (3, 50)]:
        i = hand_index(a, b)
        cs = {int(HAND_CARDS[i, 0]), int(HAND_CARDS[i, 1])}
        assert cs == {a, b}
    # conflict is symmetric, diagonal true
    assert CONFLICT[5, 5]
    assert np.array_equal(CONFLICT, CONFLICT.T)
    print("  hand indexing + conflict mask ✓")


def test_showdown_vs_bruteforce():
    board = [0, 4, 8, 20, 40]  # 2c 3c 4c 7c Qc
    ranks, valid = sd.hand_ranks(board)
    assert int(valid.sum()) == 1081  # 1326 - C(47,2)... board uses 5 cards
    sign = sd.showdown_sign_matrix(ranks, valid, board)
    rng = np.random.default_rng(0)
    opp = rng.random(NUM_HANDS) * valid
    v_fast = sd.showdown_values(sign, opp, stake=10.0)
    for h in rng.choice(np.flatnonzero(valid), 6, replace=False):
        s = 0.0
        for j in range(NUM_HANDS):
            if opp[j] == 0 or CONFLICT[h, j]:
                continue
            s += opp[j] * (1.0 if ranks[h] < ranks[j] else (-1.0 if ranks[h] > ranks[j] else 0.0))
        assert abs(s * 10.0 - v_fast[h]) < 1e-6
    print("  showdown values match brute force ✓")


def test_river_exploitability():
    for seed in (7, 11):
        sg = build_river_subgame(seed=seed, starting_stack_bb=50,
                                 allowed_actions=(0, 1, 10))
        free = board_free_mask(sg.board).astype(np.float64)
        b = free / free.sum()
        sv = CfrSolver(sg, (b.copy(), b.copy()), num_iters=800, linear=True)
        sv.multistep()
        e = exploitability_bb(sg, sv.average_strategy(), (b, b))
        assert e < 0.02, f"seed {seed}: exploitability {e}"
    print("  river CFR-D converges to ~0 exploitability ✓")


def test_pushfold_matches_oracle():
    import pokertrainer_engine as pte
    from evaluate.preflop_equity import build_equity_matrix, COMBOS
    from evaluate import pushfold_solver as ps
    from rebel_py.validate import _pushfold_subgame

    ev = pte.HandEvaluator.load_or_generate("")
    E, C = build_equity_matrix(ev, 1_500_000, seed=3)
    combos = COMBOS.astype(np.float64)
    sg = _pushfold_subgame(E, C, combos, stack_bb=10)
    beliefs = (combos / combos.sum(), combos / combos.sum())
    sv = CfrSolver(sg, beliefs, num_iters=3000, linear=True)
    sv.multistep()
    avg = sv.average_strategy()
    sb_jam, bb_call = avg[0][:, 1], avg[2][:, 1]
    oracle = ps.solve_pushfold(E, C, 10, iters=2500)
    w = combos
    sj = float((w * ((sb_jam >= 0.5) == (oracle.sb_jam >= 0.5))).sum() / w.sum())
    bc = float((w * ((bb_call >= 0.5) == (oracle.bb_call >= 0.5))).sum() / w.sum())
    assert sj > 0.9 and bc > 0.9, f"agreement SB {sj:.2f} BB {bc:.2f}"
    print(f"  push/fold reproduces oracle (SB agree {sj*100:.0f}%, BB {bc*100:.0f}%) ✓")


def main():
    print("[test_rebel_phase1] running...")
    test_hand_index()
    test_showdown_vs_bruteforce()
    test_river_exploitability()
    test_pushfold_matches_oracle()
    print("[test_rebel_phase1] all tests passed ✓")


if __name__ == "__main__":
    main()
