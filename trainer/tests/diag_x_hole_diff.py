"""Diagnostic: are the hole-card bits of obs.x actually different between
hands at the same SB-open state?

If the encoder has regressed (cf. the historical pybind 1D-array zeroing
bug from 2026-04-25), the network would have no signal to differentiate
on and the hole-card insensitivity finding is meaningless. Verify before
suggesting algorithm fixes.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/diag_x_hole_diff.py
"""
from __future__ import annotations

import numpy as np

import pokertrainer_engine as pte


def _find_seed(env, predicate, max_tries=5000) -> int:
    for s in range(max_tries):
        env.reset(s)
        if predicate(env.state().hole[0]):
            return s
    raise RuntimeError("no matching seed")


def _ranks(h):
    return tuple(sorted((int(h[0]) >> 2, int(h[1]) >> 2), reverse=True))


def main() -> None:
    env = pte.Env(0xD1A6)

    # Find an AA deal and a 72o deal.
    s_aa = _find_seed(env, lambda h: _ranks(h) == (12, 12))
    env.reset(s_aa)
    x_aa = env.observation().x.copy()
    hole_aa = list(env.state().hole[0])

    s_72o = _find_seed(env, lambda h: _ranks(h) == (5, 0)
                       and (int(h[0]) & 3) != (int(h[1]) & 3))
    env.reset(s_72o)
    x_72o = env.observation().x.copy()
    hole_72o = list(env.state().hole[0])

    print(f"AA  hole cards: {hole_aa}  (ranks {[c >> 2 for c in hole_aa]})")
    print(f"72o hole cards: {hole_72o} (ranks {[c >> 2 for c in hole_72o]})")
    print(f"x shape: {x_aa.shape}")
    print()

    # Where do the two x's differ?
    diff_mask = x_aa != x_72o
    diff_idx = np.where(diff_mask)[0]
    print(f"Number of differing dimensions: {len(diff_idx)}")
    if len(diff_idx) <= 30:
        print(f"Differing indices: {list(diff_idx)}")
        for i in diff_idx:
            print(f"  x[{i}]: AA={x_aa[i]:.3f}  72o={x_72o[i]:.3f}")

    # Reference: the encoder docstring says hole_cards is a 52-bit region.
    # Per docs/STATE_ENCODING.md, with v0.5 encoding it lives somewhere in
    # the first ~120 dims of x. If diff_idx is contained in [0, ~120) we
    # know hole cards are encoded; if it lives elsewhere or is empty, the
    # encoder regressed.
    print()
    print(f"max |x_aa - x_72o|: {float(np.abs(x_aa - x_72o).max()):.4f}")
    print(f"L2 distance        : {float(np.linalg.norm(x_aa - x_72o)):.4f}")

    # Quick check on which structural region the diffs land in (per pte
    # offsets exported from py_env.cpp):
    print()
    print(f"X_OFF_LEGAL_MASK={pte.X_OFF_LEGAL_MASK}  X_OFF_PREFLOP={pte.X_OFF_PREFLOP}")
    static = (diff_idx < pte.X_OFF_LEGAL_MASK).sum()
    legal  = ((diff_idx >= pte.X_OFF_LEGAL_MASK) &
              (diff_idx < pte.X_OFF_HIST_TRUNCATED)).sum()
    hist   = (diff_idx >= pte.X_OFF_PREFLOP).sum()
    print(f"diffs in static region  [0,{pte.X_OFF_LEGAL_MASK}):       {static}")
    print(f"diffs in legal/trunc region:                          {legal}")
    print(f"diffs in history region [{pte.X_OFF_PREFLOP},816):       {hist}")


if __name__ == "__main__":
    main()
