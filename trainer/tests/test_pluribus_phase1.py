"""Phase-1 Pluribus tests: abstraction, infoset keying, tables, MCCFR sanity.

Fast unit checks (+ one small training smoke). The full Nash-oracle gate lives
in ``pluribus.validate`` (it trains a real blueprint); here we keep it quick.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_pluribus_phase1.py
"""
from __future__ import annotations

import pluribus  # noqa: F401  (pins BLAS threads before numpy loads)
import numpy as np
import pokertrainer_engine as pte

from pluribus.config import PluribusConfig, FOLD, CHECK_CALL, ALL_IN, NUM_ACTIONS
from pluribus.abstraction import build_abstraction, N_PREFLOP_CLASSES
from pluribus.infoset import Tables, infoset_key
from pluribus.blueprint import train_blueprint, Blueprint
from evaluate.preflop_equity import class_of


def test_action_abstraction():
    abst = build_abstraction(PluribusConfig(action_preset="pushfold"))
    env = pte.Env(7, 1000); env.reset(7)            # SB to open, 10bb
    st = env.state()
    legal = [int(a) for a in st.legal_actions()]
    acts = abst.action.actions(st, legal)
    assert set(acts) == {FOLD, ALL_IN}, acts        # no limping on the open
    env.step_action(pte.ActionType(ALL_IN))         # SB jams → BB faces all-in
    st = env.state()
    acts = abst.action.actions(st, [int(a) for a in st.legal_actions()])
    assert set(acts) == {FOLD, CHECK_CALL}, acts     # call or fold
    # discrete preset force-includes the structural actions
    d = build_abstraction(PluribusConfig(action_preset="discrete"))
    assert {FOLD, CHECK_CALL, ALL_IN} <= set(d.action.preflop)
    print("  action abstraction (pushfold + discrete) ✓")


def test_info_abstraction_preflop():
    info = build_abstraction(PluribusConfig()).info
    # preflop bucket == lossless 169-class id, for many combos
    for (a, b) in [(0, 1), (48, 49), (12 * 4, 12 * 4 + 1), (3, 50)]:
        assert info.bucket_cards(0, [], a, b) == class_of(a, b)
    # AA, KK, AKs distinct; full class space present
    ids = {info.bucket_cards(0, [], a, b)
           for a in range(52) for b in range(a + 1, 52)}
    assert len(ids) == N_PREFLOP_CLASSES == 169
    print("  preflop info abstraction = 169 lossless classes ✓")


def test_infoset_key():
    env = pte.Env(7, 1000); env.reset(7)
    info = build_abstraction(PluribusConfig()).info
    k0 = infoset_key(env.state(), info)
    assert k0[0] == 0 and k0[1] in (0, 1) and isinstance(k0[3], tuple) and k0[3] == ()
    env.step_action(pte.ActionType(ALL_IN))
    k1 = infoset_key(env.state(), info)
    assert k1[3] == (ALL_IN,), k1                    # history token = abstract line
    print("  infoset key (street, to_act, bucket, history) ✓")


def test_tables():
    t = Tables(NUM_ACTIONS)
    key = (0, 0, 5, ())
    mask = np.zeros(NUM_ACTIONS); mask[[FOLD, ALL_IN]] = 1.0
    t.add_regret(key, np.where(mask > 0, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]), 0))
    # uniform fallback when no strategy accumulated
    assert np.isclose(t.average_strategy(key, mask).sum(), 1.0)
    # additive + merge
    t.add_strat(key, mask * 2.0)
    other = Tables(NUM_ACTIONS); other.add_strat(key, mask * 6.0)
    t.merge_from(other)
    avg = t.average_strategy(key, mask)
    assert np.isclose(avg[FOLD], 0.5) and np.isclose(avg[ALL_IN], 0.5)
    print("  tables: additive accumulate + merge + normalize ✓")


def test_allin_equity_payoff():
    from pluribus.mccfr import _allin_equity_payoff
    # Construct AA vs 72o all-in preflop; AA must be a heavy favorite.
    env = pte.Env(1, 1000); env.reset(1)
    st = env.state()
    AA = [[12 * 4 + 0, 12 * 4 + 1]]      # Ac Ad
    o72 = [[5 * 4 + 2, 0 * 4 + 3]]       # 7h 2s
    st.hole = AA + o72
    env.step_action(pte.ActionType(ALL_IN))   # SB jam
    env.step_action(pte.ActionType(CHECK_CALL))  # BB call → all-in showdown
    rng = np.random.default_rng(0)
    v_sb = _allin_equity_payoff(env.state(), 0, rng, 200)
    assert v_sb > 2.0, v_sb               # AA wins big on average (bb units, +)
    print(f"  all-in equity payoff (AA vs 72o → SB +{v_sb:.1f}bb) ✓")


def test_pushfold_training_smoke():
    # Small blueprint: strong hands should jam, trash should fold.
    cfg = PluribusConfig(stack_bb=10, action_preset="pushfold", iters=8000,
                         allin_equity_samples=24, seed=3, log_every=10**9)
    tables, abst = train_blueprint(cfg, progress=False)
    bp = Blueprint.from_tables(tables, abst)
    env = pte.Env(2, 1000); env.reset(2)
    st = env.state()
    def jam_freq(c0, c1):
        st.hole = [[c0, c1], [int(st.hole[1][0]), int(st.hole[1][1])]]
        return bp.strategy(st)[ALL_IN]
    aa = jam_freq(12 * 4, 12 * 4 + 1)            # AA
    trash = jam_freq(5 * 4 + 2, 0 * 4 + 3)       # 72o
    assert aa > 0.9, f"AA jam {aa:.2f}"
    assert trash < 0.5, f"72o jam {trash:.2f}"
    print(f"  pushfold training smoke (AA jam {aa*100:.0f}%, 72o {trash*100:.0f}%) ✓")


def main():
    print("[test_pluribus_phase1] running...")
    test_action_abstraction()
    test_info_abstraction_preflop()
    test_infoset_key()
    test_tables()
    test_allin_equity_payoff()
    test_pushfold_training_smoke()
    print("[test_pluribus_phase1] all tests passed ✓")


if __name__ == "__main__":
    main()
