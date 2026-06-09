"""Tests for the stage-1 push/fold machinery.

Covers:
  1. 169-class indexing and the 13×13 grid round-trip.
  2. Range-string parser (pairs, suited/offsuit "+", single hands, percentages).
  3. Monte-Carlo equity sanity (AA crushes a random range; 72o does not).
  4. Nash solver vs the published 10 bb chart (combo-weighted agreement).
  5. Action-restricted traversal: traverse_coro with allowed={F,C,AI} never
     branches or assigns regret to a forbidden slot.
  6. Validation read-out: a zero-init AdvNet reports uniform-over-legal ranges.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_pushfold.py
"""
from __future__ import annotations

import numpy as np
import torch

import pokertrainer_engine as pte

from evaluate.preflop_equity import (N_CLASSES, CLASS_LABELS, LABEL_TO_ID, COMBOS,
                                     class_of, grid_cell_class, build_equity_matrix)
from evaluate.pushfold_solver import solve_pushfold
from evaluate.pushfold_reference import (parse_range, reference_pct, compare,
                                         SB_SHOVE_10BB, BB_CALL_10BB)

NUM_ACTIONS = 11
_FOLD, _CALL, _ALL_IN = 0, 1, 10

_EQ_CACHE: dict = {}


def _equity(n_deals: int = 2_000_000):
    if "EC" not in _EQ_CACHE:
        ev = pte.HandEvaluator.load_or_generate("")
        _EQ_CACHE["EC"] = build_equity_matrix(ev, n_deals, seed=2024)
    return _EQ_CACHE["EC"]


# ─── 1. class indexing / grid ────────────────────────────────────────────────

def test_class_indexing():
    assert len(CLASS_LABELS) == N_CLASSES == 169
    assert len(set(CLASS_LABELS)) == 169                  # all distinct
    assert COMBOS.sum() == 1326                            # C(52,2)
    # class_of agrees with labels for a few explicit holes.
    assert CLASS_LABELS[class_of(12 << 2 | 0, 12 << 2 | 1)] == "AA"
    assert CLASS_LABELS[class_of(5 << 2 | 0, 0 << 2 | 1)] == "72o"
    assert CLASS_LABELS[class_of(5 << 2 | 0, 0 << 2 | 0)] == "72s"
    # Grid corners: AA top-left, 22 bottom-right, A2s top-right, A2o bottom-left.
    assert CLASS_LABELS[grid_cell_class(0, 0)] == "AA"
    assert CLASS_LABELS[grid_cell_class(12, 12)] == "22"
    assert CLASS_LABELS[grid_cell_class(0, 12)] == "A2s"
    assert CLASS_LABELS[grid_cell_class(12, 0)] == "A2o"
    print("  class indexing + grid round-trip ✓")


# ─── 2. range parser ─────────────────────────────────────────────────────────

def test_range_parser():
    assert len(parse_range("22+")) == 13
    assert len(parse_range("A2s+")) == 12                 # A2s..AKs
    assert len(parse_range("K4o+")) == 9                  # K4o..KQo
    assert parse_range("T9o") == frozenset({LABEL_TO_ID["T9o"]})
    assert parse_range("AA") == frozenset({LABEL_TO_ID["AA"]})
    # The published SB range really is ~57% of hands (string, not the often-
    # mislabeled quoted percent).
    assert 0.53 < reference_pct(SB_SHOVE_10BB) < 0.60
    assert 0.40 < reference_pct(BB_CALL_10BB) < 0.45
    print("  range parser + reference percentages ✓")


# ─── 3. equity sanity ────────────────────────────────────────────────────────

def test_equity_sanity():
    E, C = _equity()
    # Row-aggregate equity vs the whole opposing range (this is what the solver
    # uses; well-sampled even when single cells are sparse).
    def vs_range(label):
        i = LABEL_TO_ID[label]
        w = C[i].astype(float)
        return float((E[i] * w).sum() / w.sum())
    aa = vs_range("AA")
    junk = vs_range("72o")
    assert aa > 0.83, aa                                   # AA ~85% vs random
    assert junk < 0.36, junk                               # 72o ~32% vs random
    assert aa - junk > 0.45
    print(f"  equity-vs-range: AA={aa:.3f} 72o={junk:.3f} ✓")


# ─── 4. solver vs published chart ────────────────────────────────────────────

def test_solver_matches_published():
    E, C = _equity()
    orc = solve_pushfold(E, C, 10.0, iters=4000)
    # Range sizes in the published ballpark.
    assert 0.52 < orc.sb_jam_pct < 0.64, orc.sb_jam_pct
    assert 0.33 < orc.bb_call_pct < 0.45, orc.bb_call_pct
    # Combo-weighted decision agreement with published Nash (boundary hands
    # differ by design — chip-EV exact equity vs a rounded chart, and 9 vs 10bb
    # on the BB side — so the bar is "shape matches", not 100%).
    sb = compare(orc.sb_jam, parse_range(SB_SHOVE_10BB))
    bb = compare(orc.bb_call, parse_range(BB_CALL_10BB))
    assert sb["agreement"] > 0.82, sb
    assert bb["agreement"] > 0.88, bb
    # Strength monotonicity at the extremes.
    assert orc.sb_jam[LABEL_TO_ID["AA"]] > 0.99
    assert orc.sb_jam[LABEL_TO_ID["32o"]] < 0.01
    assert orc.bb_call[LABEL_TO_ID["AA"]] > 0.99
    assert orc.bb_call[LABEL_TO_ID["72o"]] < 0.01
    print(f"  solver vs published: SB agree={sb['agreement']*100:.1f}% "
          f"BB agree={bb['agreement']*100:.1f}% ✓")


# ─── 5. action-restricted traversal ──────────────────────────────────────────

def _drive_zero(coro) -> float:
    """Run a traverse_coro to completion, answering every inference request
    with all-zero regrets (⇒ uniform-over-allowed σ). Returns its value."""
    val = 0.0
    try:
        coro.send(None)
        while True:
            coro.send(np.zeros(NUM_ACTIONS, dtype=np.float32))
    except StopIteration as stop:
        val = float(stop.value) if stop.value is not None else 0.0
    return val


def test_traversal_action_restriction():
    from cfr.cfr_coro import traverse_coro
    allowed = frozenset({_FOLD, _CALL, _ALL_IN})
    rng = np.random.default_rng(0)
    forbidden = [a for a in range(NUM_ACTIONS) if a not in allowed]
    for seed in range(40):
        env = pte.Env(seed, 1000)            # 10 bb
        env.reset(seed)
        adv_writes, pol_writes = [], []
        coro = traverse_coro(env, traverser=int(rng.integers(2)), rng=rng,
                             adv_writes=adv_writes, pol_writes=pol_writes,
                             allowed=allowed)
        _drive_zero(coro)
        for tok_state, regrets in adv_writes:
            # No regret mass may land on a forbidden slot.
            assert np.all(regrets[forbidden] == 0.0), regrets
    print("  action-restricted traversal: no regret on forbidden slots ✓")


# ─── 6. validation read-out on a zero-init net ───────────────────────────────

def test_validation_zero_init_uniform():
    from cfr.config import CFRConfig
    from cfr.models import AdvNet
    from cfr.pushfold_validation import run_pushfold_validation
    from evaluate.pushfold_solver import PushFoldOracle

    cfg = CFRConfig()
    dev = torch.device("cpu")
    nets = [AdvNet(cfg.model).to(dev), AdvNet(cfg.model).to(dev)]
    orc = PushFoldOracle(10.0, np.zeros(N_CLASSES), np.zeros(N_CLASSES), 0, 0)
    rep = run_pushfold_validation(nets, dev, orc, starting_stack_chips=1000,
                                  allowed=frozenset({_FOLD, _CALL, _ALL_IN}))
    # Zero-init head ⇒ uniform σ over the 3 legal SB actions (jam=1/3) and the
    # 2 legal BB actions (call=1/2).
    assert np.allclose(rep.net_jam, 1.0 / 3.0, atol=1e-3), rep.net_jam[:3]
    assert np.allclose(rep.net_call, 0.5, atol=1e-3), rep.net_call[:3]
    print("  validation read-out on zero-init net is uniform ✓")


def main() -> None:
    print("[test_pushfold] running...")
    test_class_indexing()
    test_range_parser()
    test_equity_sanity()
    test_solver_matches_published()
    test_traversal_action_restriction()
    test_validation_zero_init_uniform()
    print("[test_pushfold] all tests passed ✓")


if __name__ == "__main__":
    main()
