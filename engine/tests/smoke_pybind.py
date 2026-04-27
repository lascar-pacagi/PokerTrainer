"""Smoke test: drive the C++ Env from Python.

Run with
    PYTHONPATH=engine/build python engine/tests/smoke_pybind.py
from the PokerTrainer repo root.
"""
import sys
import numpy as np

import pokertrainer_engine as pte

print(f"module version check — X_DIM={pte.X_DIM}, A_DIM={pte.A_DIM}, "
      f"HIST_MAX={pte.HIST_MAX}, HIST_FEAT={pte.HIST_FEAT}, "
      f"NUM_ACTIONS={pte.NUM_ACTIONS}")
assert pte.X_DIM == 745
assert pte.A_DIM == 17
assert pte.HIST_MAX == 24
assert pte.HIST_FEAT == 25
assert pte.NUM_ACTIONS == 11
assert pte.STATIC_DIM == 121
assert pte.X_OFF_HIST == 121
assert pte.X_OFF_VALID_MASK == 721

# One complete hand: check-call all the way to showdown.
env = pte.Env(seed=2026)
assert not env.is_terminal()
assert env.to_act() == pte.Player.SB

total_steps = 0
while not env.is_terminal():
    obs = env.observation()
    # Shape checks.
    assert obs.x.shape == (pte.X_DIM,) and obs.x.dtype == np.float32
    assert obs.a.shape == (len(obs.legal), pte.A_DIM)
    assert obs.legal_idx.shape == (len(obs.legal),)
    # Valid-mask only 1.0 for as many actions as have happened so far,
    # saturating at HIST_MAX.
    n_valid = int(obs.x[pte.X_OFF_VALID_MASK:pte.X_OFF_VALID_MASK + pte.HIST_MAX].sum())
    hist_n = env.state().history_size
    assert n_valid == min(hist_n, pte.HIST_MAX), (n_valid, hist_n)

    # Preflop sanity: exactly 2 hole-card bits, 0 board-card bits.
    if env.state().street == pte.Street.PREFLOP:
        assert obs.x[:52].sum() == 2.0
        assert obs.x[52:104].sum() == 0.0

    # Prefer CHECK_CALL; otherwise fold.
    legal = list(obs.legal)
    idx = legal.index(pte.ActionType.CHECK_CALL) if pte.ActionType.CHECK_CALL in legal else 0
    result = env.step(idx)
    total_steps += 1
    if result.done:
        print(f"hand done in {total_steps} steps, payoffs = {env.payoffs_bb()}")

pays = env.payoffs_bb()
assert pays[0] + pays[1] == 0.0

# Reproducibility: same seed → same hole cards.
a = pte.Env(seed=1); a.reset(42)
b = pte.Env(seed=999); b.reset(42)
assert a.state().hole == b.state().hole
assert a.state().board == b.state().board

# HandEvaluator still works through the exposed API.
e = pte.HandEvaluator.load_or_generate("")
royal = [pte.parse_card(s) for s in ("As", "Ks", "Qs", "Js", "Ts", "2c", "3d")]
r = e.evaluate7(royal)
assert r == 1, f"royal flush should be rank 1, got {r}"
assert pte.HandEvaluator.rank_category(r) == "Straight Flush"

print("OK — all smoke checks passed")
