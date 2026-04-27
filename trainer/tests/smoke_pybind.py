"""Pybind boundary smoke — would have caught the 2026-04-25 zero-x bug.

The C++ Catch2 suite (37 tests) doesn't exercise the pybind11 layer. This file
covers the gap: every property exposed on `pte.EncodedState` must round-trip
correctly to Python. Specifically, every fresh-deal preflop observation must
have non-zero `x`, populated hole-card bits, a one-hot street at x[104], and
a legal_idx that matches `legal`.

Run:
    PYTHONPATH=engine/build:trainer python -m tests.smoke_pybind
or:
    PYTHONPATH=engine/build:trainer python trainer/tests/smoke_pybind.py
"""
from __future__ import annotations

import sys

import numpy as np

import pokertrainer_engine as pte


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  ok:   {msg}")

    print("[pybind smoke] preflop observation invariants")
    for seed in (0, 1, 42, 0xC0FFEE):
        env = pte.Env(seed)
        env.reset()
        obs = env.observation()
        x = np.asarray(obs.x)
        a = np.asarray(obs.a)
        legal = list(obs.legal)
        legal_idx = np.asarray(obs.legal_idx)

        # 1. obs.x must not be silently zero (the canonical pybind 1D-array bug).
        check(x.sum() > 0,
              f"seed={seed:#x}: obs.x.sum() > 0 (got {x.sum():.1f})")

        # 2. Hole cards are one-hot in x[0:52].
        sb_hole = list(env.state().hole[0])
        for c in sb_hole:
            check(x[c] == 1.0,
                  f"seed={seed:#x}: x[{c}] (SB hole) == 1.0 (got {x[c]})")

        # 3. Preflop one-hot at x[104] (street offsets 104..107 = pre/flop/turn/river).
        check(x[104] == 1.0,
              f"seed={seed:#x}: x[104] (preflop bit) == 1.0 (got {x[104]})")
        check(x[105] == x[106] == x[107] == 0.0,
              f"seed={seed:#x}: only the preflop bit set on a fresh deal")

        # 4. Position one-hot at x[108]/x[109]. In HU, SB acts FIRST preflop
        #    (so it's OOP this street; it'll be IP on later streets). x[108]=OOP.
        check(x[108] == 1.0 and x[109] == 0.0,
              f"seed={seed:#x}: position bits OOP=1, IP=0 for preflop SB")

        # 5. legal_idx mirrors legal exactly (this property had the same bug).
        check(len(legal) == len(legal_idx),
              f"seed={seed:#x}: len(legal_idx)==len(legal) ({len(legal_idx)}=={len(legal)})")
        for i, (lt, li) in enumerate(zip(legal, legal_idx)):
            check(int(lt) == int(li),
                  f"seed={seed:#x}: legal[{i}]={int(lt)} == legal_idx[{i}]={int(li)}")

        # 6. Per-action vectors are non-trivial: each row has its one-hot bit + scalars.
        check(a.shape == (len(legal), pte.A_DIM),
              f"seed={seed:#x}: obs.a.shape == ({len(legal)}, {pte.A_DIM})")
        check((a[:, :pte.NUM_ACTIONS].sum(axis=1) == 1.0).all(),
              f"seed={seed:#x}: each a-row has exactly one one-hot action bit")

    print()
    if failures:
        print(f"[pybind smoke] {len(failures)} FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[pybind smoke] OK — pybind boundary intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
