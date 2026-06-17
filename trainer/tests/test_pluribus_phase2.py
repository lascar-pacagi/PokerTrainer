"""Phase-2 Pluribus tests: continuation-strategy math, Bayes ranges, re-solve.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_pluribus_phase2.py
"""
from __future__ import annotations

import pluribus  # noqa: F401  (pins BLAS threads before numpy loads)
import numpy as np
import pokertrainer_engine as pte

from pluribus.config import PluribusConfig, FOLD, CHECK_CALL, ALL_IN, NUM_ACTIONS
from pluribus.abstraction import build_abstraction
from pluribus.blueprint import train_blueprint, Blueprint
from pluribus.continuations import bias_strategy, ContinuationSet, CONTINUATIONS
from pluribus.ranges import bayes_ranges
from pluribus.search import build_engine_subgame, subgame_exploitability, _root_action_probs
from rebel_py.subgame_solver import CfrSolver
from rebel_py.hand_index import board_free_mask, hand_index


def test_bias_strategy():
    sigma = np.zeros(NUM_ACTIONS); sigma[[FOLD, CHECK_CALL, ALL_IN]] = [0.5, 0.3, 0.2]
    assert np.allclose(bias_strategy(sigma, "blueprint", 5.0), sigma)
    f = bias_strategy(sigma, "fold", 5.0)
    assert np.isclose(f.sum(), 1.0) and f[FOLD] > sigma[FOLD]      # fold boosted
    r = bias_strategy(sigma, "raise", 5.0)
    assert r[ALL_IN] > sigma[ALL_IN] and np.isclose(r.sum(), 1.0)  # raise boosted
    # support preserved (no action zeroed → beliefs never collapse)
    assert (f[[FOLD, CHECK_CALL, ALL_IN]] > 0).all()
    print("  continuation bias math (fold/call/raise/blueprint) ✓")


def test_continuation_set():
    cfg = PluribusConfig(stack_bb=10, action_preset="pushfold", iters=2000,
                         allin_equity_samples=8, seed=5, log_every=10**9)
    tables, abst = train_blueprint(cfg, progress=False)
    bp = Blueprint.from_tables(tables, abst)
    cs = ContinuationSet(bp, factor=5.0)
    assert len(cs) == len(CONTINUATIONS) == 4
    env = pte.Env(7, 1000); env.reset(7)
    for k in range(len(cs)):
        s = cs.sigma(k, env.state())
        assert np.isclose(s.sum(), 1.0) and (s >= 0).all()
    print("  ContinuationSet: 4 valid σ-providers ✓")


def test_bayes_ranges():
    cfg = PluribusConfig(stack_bb=10, action_preset="pushfold", iters=6000,
                         allin_equity_samples=16, seed=7, log_every=10**9)
    tables, abst = train_blueprint(cfg, progress=False)
    bp = Blueprint.from_tables(tables, abst)
    env = pte.Env(7, 1000); env.reset(7)
    env.step_action(pte.ActionType(ALL_IN))     # SB jams
    r_sb, r_bb = bayes_ranges(env.state(), bp)
    assert np.isclose(r_sb.sum(), 1.0) and np.isclose(r_bb.sum(), 1.0)
    # After a jam, the SB range should weight AA more than 72o.
    aa = hand_index(12 * 4, 12 * 4 + 1)
    o72 = hand_index(5 * 4 + 2, 0 * 4 + 3)
    assert r_sb[aa] > r_sb[o72], (r_sb[aa], r_sb[o72])
    print(f"  Bayes ranges (post-jam SB: AA mass {r_sb[aa]:.2e} > 72o {r_sb[o72]:.2e}) ✓")


def test_river_resolve():
    abst = build_abstraction(PluribusConfig(action_preset="discrete"))
    env = pte.Env(7, 2000); env.reset(7)
    while not env.is_terminal() and int(env.state().street) < 3:
        legal = [int(a) for a in env.state().legal_actions()]
        env.step(legal.index(CHECK_CALL))
    board = [int(c) for c in env.state().board][:5]
    sg = build_engine_subgame(env, abst, depth_limit=1)
    free = board_free_mask(board).astype(float)
    ranges = (free / free.sum(), free / free.sum())
    sv = CfrSolver(sg, ranges, num_iters=500, linear=True); sv.multistep()
    avg = sv.average_strategy()
    e = subgame_exploitability(sg, avg, ranges)
    assert e < 0.05, f"river exploitability {e:.4f}"
    # root strategy: probability mass sums to ~1 per (board-free) hand
    probs = _root_action_probs(sg, avg)
    s = probs[free > 0].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-6)
    print(f"  river re-solve (Trick 2): exploitability {e:.4f} bb/hand ✓")


def main():
    print("[test_pluribus_phase2] running...")
    test_bias_strategy()
    test_continuation_set()
    test_bayes_ranges()
    test_river_resolve()
    print("[test_pluribus_phase2] all tests passed ✓")


if __name__ == "__main__":
    main()
