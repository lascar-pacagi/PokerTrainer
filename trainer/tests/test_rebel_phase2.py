"""Phase-2a ReBeL tests: PBS query, re-rooting, value-net leaf machinery.

Structural correctness (fast). The end-to-end "exploitability drops as the net
trains" gate lives in rebel_py/train.py (it is a minutes-long training run, not
a unit test); a short convergence smoke is included at the bottom and is the
one slow check here.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_rebel_phase2.py
"""
from __future__ import annotations

import rebel_py  # noqa: F401  (pins BLAS threads before numpy loads)
import numpy as np

from rebel_py.pbs import (PublicState, encode_query, decode_query, query_dim)
from rebel_py.public_tree import build_river_subgame, subtree_subgame
from rebel_py.subgame_solver import CfrSolver
from rebel_py.exploitability import exploitability_bb
from rebel_py.hand_index import board_free_mask, NUM_HANDS
from rebel_py.recursive import recursive_strategy, SubgameParams, uniform_beliefs


def test_query_roundtrip():
    ps = PublicState(street=3, board=[0, 4, 8, 20, 40], to_act=1,
                     pot_bb=13.0, to_call_bb=11.0, stack_bb=11.0)
    rng = np.random.default_rng(0)
    r0 = rng.random(NUM_HANDS); r1 = rng.random(NUM_HANDS)
    q = encode_query(1, ps, r0, r1)
    assert q.shape == (query_dim(),)
    trav, ps2, b0, b1 = decode_query(q)
    assert trav == 1 and ps2.to_act == 1 and ps2.street == 3
    assert sorted(ps2.board) == sorted(ps.board)
    assert abs(ps2.pot_bb - 13.0) < 1e-3 and abs(ps2.to_call_bb - 11.0) < 1e-3
    # reaches come back normalized
    assert abs(b0.sum() - 1.0) < 1e-6 and abs(b1.sum() - 1.0) < 1e-6
    print("  PBS query encode/decode roundtrip ✓")


def test_subtree_rerooting():
    full = build_river_subgame(seed=7, starting_stack_bb=12, allowed_actions=(0, 1, 10))
    # re-rooting at node 0 with no depth limit reproduces the full tree
    dl = subtree_subgame(full, 0, None)
    assert len(dl.nodes) == len(full.nodes)
    assert sum(nd.is_net_leaf for nd in dl.nodes) == 0
    b = uniform_beliefs(full.board)
    e0 = exploitability_bb(full, _solve(full, b), b)
    e1 = exploitability_bb(dl, _solve(dl, b), b)
    assert abs(e0 - e1) < 1e-3
    # a depth limit introduces net leaves
    dl2 = subtree_subgame(full, 0, 2)
    assert sum(nd.is_net_leaf for nd in dl2.nodes) >= 1
    for nd in dl2.nodes:
        if nd.is_net_leaf:
            assert nd.public_state is not None
    print("  subtree re-rooting + depth-limit net leaves ✓")


def _solve(sg, b, iters=500):
    sv = CfrSolver(sg, (b[0].copy(), b[1].copy()), num_iters=iters, linear=True)
    sv.multistep()
    return sv.average_strategy()


def test_recursive_assembly_exact():
    """Consistency gate: `recursive_strategy` with no net leaves (depth ≥ tree
    depth) must reproduce the direct full-solve exploitability. The naive
    "re-solve every node independently" assembly fails this (it is unsafe and
    leaves ~0.1 bb/hand exploitable even with exact solves)."""
    full = build_river_subgame(seed=1, starting_stack_bb=12, allowed_actions=(0, 1, 10))
    b = uniform_beliefs(full.board)
    e_direct = exploitability_bb(full, _solve(full, b, iters=1000), b)

    def no_net(queries):
        return np.zeros((len(queries), NUM_HANDS))

    for depth in (10, 3):  # 12bb F/C/AI river is <3 deep → no net leaves
        strat = recursive_strategy(full, no_net,
                                   SubgameParams(depth_limit=depth, num_iters=1000,
                                                 random_action_prob=0.0))
        e = exploitability_bb(full, strat, b)
        assert abs(e - e_direct) < 5e-3, f"depth {depth}: {e} vs direct {e_direct}"
    print("  recursive assembly (exact) matches direct solve ✓")


def test_net_leaf_scaling():
    """A constant value fn that returns v per (normalized) hand must yield leaf
    counterfactual values = v * opp_reach_mass (the ReBeL leaf-scaling trick)."""
    full = build_river_subgame(seed=7, starting_stack_bb=12, allowed_actions=(0, 1, 10))
    dl = subtree_subgame(full, 0, 2)
    valid = board_free_mask(full.board).astype(np.float64)
    const = (0.3 * valid)

    def vfn(queries):
        return np.tile(const, (len(queries), 1))

    b = uniform_beliefs(full.board)
    sv = CfrSolver(dl, (b[0].copy(), b[1].copy()), num_iters=1, linear=True, value_fn=vfn)
    sv.step(0)  # traverser 0
    # find a net leaf, check value == const * opp_reach_mass
    leaf = next(nid for nid, nd in enumerate(dl.nodes) if nd.is_net_leaf)
    mass = sv.reach[1][leaf].sum()  # opp (player 1) reach mass
    expected = const * valid * mass
    assert np.allclose(sv.values[leaf], expected, atol=1e-9)
    print("  net-leaf value = net_output × opp_reach_mass ✓")


def test_training_reduces_exploitability():
    """Short self-play training drives the recursive strategy's exploitability
    well below the untrained (≈near-zero value fn) baseline."""
    import torch
    from rebel_py.models import ValueNet, NetValueFn
    from rebel_py.replay import QueryValueBuffer
    from rebel_py.recursive import RlRunner

    torch.manual_seed(0)
    full = build_river_subgame(seed=2, starting_stack_bb=10, allowed_actions=(0, 1, 10))
    qdim = query_dim()
    net = ValueNet(qdim, n_hidden=128, n_layers=2)
    vfn = NetValueFn(net, "cpu")
    buf = QueryValueBuffer(60_000, qdim, NUM_HANDS)
    params = SubgameParams(depth_limit=2, num_iters=64, random_action_prob=0.25)
    rng = np.random.default_rng(0)
    b = uniform_beliefs(full.board)

    def exploit():
        s = recursive_strategy(full, vfn, SubgameParams(depth_limit=2, num_iters=96,
                                                        random_action_prob=0.0))
        return exploitability_bb(full, s, b)

    e_before = exploit()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    lossf = torch.nn.SmoothL1Loss()
    for _ in range(16):
        net.eval()
        for _ in range(24):
            RlRunner(full, vfn, params, buf, rng).step()
        net.train()
        for _ in range(48):
            q, v = buf.sample(256)
            loss = lossf(net(torch.tensor(q)), torch.tensor(v))
            opt.zero_grad(); loss.backward(); opt.step()
    e_after = exploit()
    print(f"  training: exploit {e_before:.4f} → {e_after:.4f} bb/hand")
    # The mechanism gate: self-play + value-net learning substantially reduces
    # the recursive strategy's exploitability. (Driving it to the exact floor is
    # a net-quality / compute matter handled at Phase-3 scale.)
    assert e_after < 0.4 * e_before, (e_before, e_after)
    print("  self-play training reduces exploitability ✓")


def main():
    print("[test_rebel_phase2] running...")
    test_query_roundtrip()
    test_subtree_rerooting()
    test_net_leaf_scaling()
    test_training_reduces_exploitability()
    print("[test_rebel_phase2] all tests passed ✓")


if __name__ == "__main__":
    main()
