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
assert pte.X_DIM == 816
assert pte.A_DIM == 11
assert pte.HIST_MAX == 34
assert pte.HIST_FEAT == 20
assert pte.NUM_ACTIONS == 11
assert pte.STATIC_DIM == 136
assert pte.LEGAL_MASK_DIM == 11
assert pte.HIST_TRUNC_DIM == 4
assert pte.X_OFF_LEGAL_MASK == 121
assert pte.X_OFF_HIST_TRUNCATED == 132
assert pte.X_OFF_PREFLOP == 136
assert pte.X_OFF_FLOP    == 136 + 10 * pte.HIST_FEAT
assert pte.X_OFF_TURN    == pte.X_OFF_FLOP + 8 * pte.HIST_FEAT
assert pte.X_OFF_RIVER   == pte.X_OFF_TURN + 8 * pte.HIST_FEAT
assert pte.STREET_SLOTS   == (10, 8, 8, 8)
assert pte.STREET_OFFSETS == (pte.X_OFF_PREFLOP, pte.X_OFF_FLOP, pte.X_OFF_TURN, pte.X_OFF_RIVER)

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
    # Fixed-position history: the count of populated rows in each street's
    # sub-block must equal the count of history actions on that street so
    # far. Sum across sub-blocks equals total history size (bounded by the
    # per-street slot budgets, which we never exceed in this smoke).
    n_real_per_street = []
    for street_idx, off in enumerate(pte.STREET_OFFSETS):
        slots = pte.STREET_SLOTS[street_idx]
        is_real_offsets = off + np.arange(slots) * pte.HIST_FEAT
        n_real_per_street.append(int(obs.x[is_real_offsets].sum()))
    n_real_total = sum(n_real_per_street)
    hist_n = env.state().history_size
    assert n_real_total == hist_n, (n_real_per_street, hist_n)

    # Legal-actions mask block matches the legal list.
    legal_set = {int(a) for a in obs.legal}
    for k in range(pte.LEGAL_MASK_DIM):
        bit = obs.x[pte.X_OFF_LEGAL_MASK + k]
        expected = 1.0 if k in legal_set else 0.0
        assert bit == expected, (k, bit, expected)

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


# ─── state.history + AppliedAction bindings ─────────────────────────────────
# Replay a few actions and assert the history vector is exposed correctly.
env = pte.Env(seed=4242)
assert list(env.state().history) == []
assert list(env.state().starting_stacks) == [10000, 10000]   # 100bb default

env.step_action(pte.ActionType.RAISE_100)   # SB opens
env.step_action(pte.ActionType.CHECK_CALL)  # BB calls

h = list(env.state().history)
assert len(h) == 2
assert h[0].actor == pte.Player.SB
assert h[0].street == pte.Street.PREFLOP
assert h[0].type == pte.ActionType.RAISE_100
assert h[1].actor == pte.Player.BB
assert h[1].type == pte.ActionType.CHECK_CALL
# pot grew, stack shrank — basic invariants
assert h[0].pot_after_chips == 400      # 0.5 SB + 1 BB + 2 SB-raise-to-3 = pot 4bb after R100
assert h[0].stack_after_chips == 9700
assert not h[0].was_all_in
print(f"history bindings OK: {len(h)} entries, "
      f"final pot_after={h[-1].pot_after_chips}")

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
