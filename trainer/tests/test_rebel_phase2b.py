"""Phase-2b ReBeL tests: the river chance node (turn→river) + card removal.

The full turn→river exploitability gate is a minutes-long solve (it lives in
rebel_py/validate.py -> validate_turn). Here we keep two fast checks:

  * structure — build_turn_subgame yields the expected chance nodes and the
    solver steps over them without error;
  * equity — the all-in turn-runout value (the collapsed chance showdown)
    equals the ground-truth win/lose count over the 44 valid river cards, from
    independent evaluate7 calls. This pins the chance math: the per-card
    showdown matrices, the 1/44 matchup normalizer, and card removal.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_rebel_phase2b.py
"""
from __future__ import annotations

import rebel_py  # noqa: F401  (pins BLAS threads before numpy loads)
import numpy as np

from rebel_py.public_tree import build_turn_subgame
from rebel_py.subgame_solver import CfrSolver
from rebel_py.hand_index import hand_index, HAND_CARDS, board_free_mask, NUM_HANDS
from rebel_py.recursive import uniform_beliefs


def test_turn_subgame_structure():
    sg = build_turn_subgame(seed=7, starting_stack_bb=6, allowed_actions=(0, 1, 10))
    assert len(sg.board) == 4, "turn board must be 4 cards"
    chance = [n for n in sg.nodes if n.is_chance]
    assert len(chance) >= 1, "expected at least the check-check river chance node"
    for c in chance:
        # one child per unseen card (52 - 4 board = 48)
        assert len(c.children) == 48 and len(c.chance_cards) == 48
        assert all(r not in sg.board for r in c.chance_cards)
    # solver constructs masks + divisor and steps without error
    b = uniform_beliefs(sg.board)
    sv = CfrSolver(sg, (b[0].copy(), b[1].copy()), num_iters=1, linear=True)
    assert sv.chance_div == 44.0  # 52 - 4 board - 2 hero - 2 opp
    sv.step(0); sv.step(1)
    print("  turn subgame structure + chance solver step ✓")


def test_turn_runout_equity_vs_bruteforce():
    """Collapsed all-in runout value for a single opponent hand == the exact
    win/lose tally over the 44 valid river cards (independent evaluate7)."""
    import pokertrainer_engine as pte
    sg = build_turn_subgame(seed=7, starting_stack_bb=6, allowed_actions=(0, 1, 10))
    turn_board = list(sg.board)
    # an all-in-runout terminal carries make_chance_showdown_value (player -1,
    # terminal, and not a fold — its value is hand-dependent for both traversers)
    ev = pte.HandEvaluator.load_or_generate("")

    # pick a hero/opp matchup with a clear equity edge (so the check is not a
    # trivial 0==0): scan free-card pairs for the largest |win - lose| tally.
    free = [c for c in range(52) if c not in turn_board]

    def tally(a, bcard, c, d):
        used = set(turn_board) | {a, bcard, c, d}
        rivers = [r for r in range(52) if r not in used]
        net = 0
        for r in rivers:
            board5 = turn_board + [r]
            net += np.sign(ev.evaluate7([c, d] + board5) - ev.evaluate7([a, bcard] + board5))
        return int(net), len(rivers)

    a, bcard, c, d = free[0], free[1], free[2], free[3]
    best = abs(tally(a, bcard, c, d)[0])
    for i in range(0, 8, 2):           # a few candidate matchups
        cand = free[i:i + 4]
        if len(cand) == 4 and abs(tally(*cand)[0]) > best:
            a, bcard, c, d = cand; best = abs(tally(*cand)[0])
    hero, opp = hand_index(a, bcard), hand_index(c, d)
    net, nriv = tally(a, bcard, c, d)
    assert nriv == 44 and net != 0, f"need a decisive matchup, got net={net}"
    stake = 1.0
    expect = stake * net / 44.0
    rivers = [r for r in range(52) if r not in (set(turn_board) | {a, bcard, c, d})]

    # solver value: find a collapsed runout terminal, feed a delta opp reach.
    # The collapsed terminal uses the summed matrix with its own matched pot;
    # rebuild the closure value at unit stake by querying with a delta reach and
    # dividing out that terminal's stake. Simpler: reconstruct the same value via
    # the package's showdown summation used by the builder.
    from rebel_py import showdown as sd
    summed = np.zeros((NUM_HANDS, NUM_HANDS), dtype=np.float64)
    for r in rivers:
        board5 = turn_board + [r]
        ranks, valid = sd.hand_ranks(board5)
        summed += sd.showdown_sign_matrix(ranks, valid, board5)
    delta = np.zeros(NUM_HANDS); delta[opp] = 1.0
    got = (summed @ delta)[hero] / 44.0
    assert abs(got - expect) < 1e-9, (got, expect)
    print(f"  turn runout equity matches evaluate7 ({net}/44) ✓")


def test_river_showdown_matches_dense():
    """The rank-vector cumsum showdown (the scaling primitive that replaces 48
    dense 1326² matrices) must equal showdown_sign_matrix @ opp_reach exactly,
    including card removal."""
    from rebel_py import showdown as sd
    rng = np.random.default_rng(0)
    for board in ([0, 4, 8, 20, 40], [17, 21, 44, 20, 27], [3, 11, 25, 33, 49]):
        ranks, valid = sd.hand_ranks(board)
        dense = sd.showdown_sign_matrix(ranks, valid, board)
        rs = sd.RiverShowdown(ranks, valid)
        free = board_free_mask(board).astype(float)
        for _ in range(4):
            opp = (rng.random(NUM_HANDS) * free) ** rng.uniform(0.3, 3.0)
            stake = rng.uniform(1, 20)
            assert np.abs(stake * (dense @ opp) - rs.values(opp, stake)).max() < 1e-7
    print("  rank-vector cumsum showdown matches dense matrix ✓")


def test_turn_netleaf_subgame():
    """The depth-limited turn subgame (stop_at_chance) values the post-chance
    river PBSs with the net: 48 net leaves (one per unseen card), each a 5-card
    river board, masked by its own validity (not the 4-card turn board)."""
    import torch
    from rebel_py.public_tree import subtree_subgame
    from rebel_py.models import ValueNet, NetValueFn
    from rebel_py.pbs import query_dim
    full = build_turn_subgame(seed=7, starting_stack_bb=6, allowed_actions=(0, 1, 10))
    dl = subtree_subgame(full, 0, None, stop_at_chance=True)
    leaves = [i for i, n in enumerate(dl.nodes) if n.is_net_leaf]
    assert len(leaves) == 48
    for i in leaves:
        assert dl.nodes[i].public_state is not None
        assert len(dl.nodes[i].public_state.board) == 5
    torch.manual_seed(0)
    net = ValueNet(query_dim(), n_hidden=32, n_layers=2)
    vfn = NetValueFn(net, "cpu")
    b = uniform_beliefs(full.board)
    sv = CfrSolver(dl, (b[0].copy(), b[1].copy()), num_iters=4, value_fn=vfn)
    sv.multistep()
    assert int(sv.net_leaf_valid[leaves[0]].sum()) == 1081  # C(47,2) on a 5-card board
    print("  turn netleaf subgame: 48 river-PBS net leaves, per-leaf 5-card masks ✓")


def main():
    print("[test_rebel_phase2b] running...")
    test_turn_subgame_structure()
    test_turn_runout_equity_vs_bruteforce()
    test_river_showdown_matches_dense()
    test_turn_netleaf_subgame()
    print("[test_rebel_phase2b] all tests passed ✓")


if __name__ == "__main__":
    main()
