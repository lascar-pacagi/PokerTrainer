"""Heads-up Nash push/fold solver — the stage-1 training oracle.

═══════════════════════════════════════════════════════════════════════════════
THE GAME
═══════════════════════════════════════════════════════════════════════════════

At a short effective stack S (bb), HU NLHE preflop collapses to push/fold:
the SB (button, acts first) either JAMS all-in or FOLDS; if SB jams, the BB
either CALLS (all-in showdown) or FOLDS. Blinds: SB posts 0.5 bb, BB posts 1 bb.

Net payoffs to the SB, in bb, relative to the start of the hand:

    SB fold                 :  -0.5            (loses the small blind)
    SB jam, BB fold         :  +1.0            (wins the big blind)
    SB jam, BB call (eq e)  :  (2e - 1)·S      (all-in for S each; e = SB equity)

So the indifference thresholds are EV_fold(SB) = -0.5 and EV_fold(BB) = -1.0.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS IS THE RIGHT ORACLE (oracle "A")
═══════════════════════════════════════════════════════════════════════════════

Rather than hard-code a published chart whose blind/stack/rake assumptions may
differ from our engine, we SOLVE the equilibrium under our exact rules using the
engine's own showdown equities (preflop_equity.py). The result is self-consistent
with the game the CFR trainer is actually solving at stage 1. We separately
sanity-check the solved SB jam range against a published 10 bb chart
(pushfold_reference.py) to catch gross errors.

Solved by fictitious play: each player best-responds to the opponent's *average*
strategy; the running averages converge to the (zero-sum) Nash equilibrium, with
indifferent hands settling at mixed (fractional) frequencies. Everything is
vectorised over the 169×169 equity/joint-count matrices, so a solve is milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preflop_equity import (N_CLASSES, CLASS_LABELS, COMBOS,
                             load_or_build_equity, grid_cell_class)


# ═══════════════════════════════════════════════════════════════════════════
# BEST-RESPONSE STEPS (vectorised over the 169 classes)
# ═══════════════════════════════════════════════════════════════════════════


def _sb_best_response(E: np.ndarray, C: np.ndarray, bb_call: np.ndarray,
                      stack_bb: float, sb_blind: float) -> np.ndarray:
    """0/1 jam decision per SB class, given BB's calling frequencies.

    EV_jam_i = [ Σ_j C_ij(1-call_j)·(+1)  +  Σ_j C_ij·call_j·(2E_ij-1)S ] / Σ_j C_ij
    SB jams class i iff EV_jam_i ≥ EV_fold = -sb_blind.
    """
    Ni = C.sum(axis=1)                                   # combos with SB=i
    fold_term = (C * (1.0 - bb_call)[None, :]).sum(axis=1) * 1.0
    call_term = (C * bb_call[None, :] * (2.0 * E - 1.0) * stack_bb).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ev_jam = np.where(Ni > 0, (fold_term + call_term) / Ni, -np.inf)
    return (ev_jam >= -sb_blind).astype(np.float64)


def _bb_best_response(E: np.ndarray, C: np.ndarray, sb_jam: np.ndarray,
                      stack_bb: float, bb_blind: float) -> np.ndarray:
    """0/1 call decision per BB class, given SB's jamming frequencies.

    Facing a jam, BB's equity vs SB class i is (1 - E_ij); call net = (1-2E_ij)S.
    EV_call_j = Σ_i C_ij·jam_i·(1-2E_ij)S / Σ_i C_ij·jam_i
    BB calls class j iff EV_call_j ≥ EV_fold = -bb_blind.
    """
    Wj = C * sb_jam[:, None]                             # weight by SB jam freq
    Mj = Wj.sum(axis=0)                                  # combos facing a jam
    num = (Wj * (1.0 - 2.0 * E) * stack_bb).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ev_call = np.where(Mj > 0, num / Mj, -np.inf)
    return (ev_call >= -bb_blind).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PushFoldOracle:
    stack_bb: float
    sb_jam: np.ndarray      # (169,) jam frequency per class (SB open)
    bb_call: np.ndarray     # (169,) call frequency per class (BB vs a jam)
    n_deals: int            # equity-MC budget the oracle was solved from
    seed: int

    # ── Combo-weighted range sizes (fraction of all dealt hands) ────────────
    @property
    def sb_jam_pct(self) -> float:
        return float((self.sb_jam * COMBOS).sum() / COMBOS.sum())

    @property
    def bb_call_pct(self) -> float:
        return float((self.bb_call * COMBOS).sum() / COMBOS.sum())

    def jam_grid(self) -> np.ndarray:
        return _to_grid(self.sb_jam)

    def call_grid(self) -> np.ndarray:
        return _to_grid(self.bb_call)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, stack_bb=self.stack_bb, sb_jam=self.sb_jam,
                 bb_call=self.bb_call, n_deals=self.n_deals, seed=self.seed)

    @staticmethod
    def load(path: str | Path) -> "PushFoldOracle":
        with np.load(path) as z:
            return PushFoldOracle(float(z["stack_bb"]), z["sb_jam"], z["bb_call"],
                                  int(z["n_deals"]), int(z["seed"]))


def _to_grid(freq: np.ndarray) -> np.ndarray:
    """Reshape a 169-vector into the standard 13×13 grid (rows/cols A→2)."""
    g = np.zeros((13, 13), dtype=freq.dtype)
    for r in range(13):
        for c in range(13):
            g[r, c] = freq[grid_cell_class(r, c)]
    return g


# ═══════════════════════════════════════════════════════════════════════════
# SOLVE
# ═══════════════════════════════════════════════════════════════════════════


def solve_pushfold(E: np.ndarray, C: np.ndarray, stack_bb: float,
                   *, sb_blind: float = 0.5, bb_blind: float = 1.0,
                   iters: int = 4000, seed: int = 0,
                   n_deals: int = 0) -> PushFoldOracle:
    """Fictitious-play solve of the HU push/fold equilibrium.

    Both players best-respond to the opponent's running average; the averages
    converge to Nash. Indifferent hands settle at mixed frequencies.
    """
    C = C.astype(np.float64)
    sb_avg = np.full(N_CLASSES, 0.5)
    bb_avg = np.full(N_CLASSES, 0.5)
    for t in range(1, iters + 1):
        sb_br = _sb_best_response(E, C, bb_avg, stack_bb, sb_blind)
        bb_br = _bb_best_response(E, C, sb_avg, stack_bb, bb_blind)
        sb_avg += (sb_br - sb_avg) / t
        bb_avg += (bb_br - bb_avg) / t
    return PushFoldOracle(stack_bb=stack_bb, sb_jam=sb_avg, bb_call=bb_avg,
                          n_deals=n_deals, seed=seed)


def build_oracle(stack_bb: float = 10.0,
                 n_deals: int = 12_000_000,
                 seed: int = 0xE119,
                 cache_dir: str | None = None,
                 iters: int = 4000,
                 progress: bool = True) -> PushFoldOracle:
    """Load/build the equity matrix, solve, and cache the solved oracle.

    Cached under <cache_dir>/pushfold_oracle_<stack>bb_n<deals>.npz so the
    training loop pays the (~minute) equity build at most once.
    """
    from .preflop_equity import os as _os  # reuse the same default cache dir
    cache_dir = cache_dir or _os.environ.get("PT_ORACLE_CACHE", "runs/oracle_cache")
    opath = Path(cache_dir) / f"pushfold_oracle_{stack_bb:g}bb_n{n_deals}.npz"
    if opath.exists():
        return PushFoldOracle.load(opath)
    E, C = load_or_build_equity(n_deals, seed=seed, cache_dir=cache_dir,
                                progress=progress)
    oracle = solve_pushfold(E, C, stack_bb, iters=iters, seed=seed, n_deals=n_deals)
    oracle.save(opath)
    if progress:
        print(f"  [oracle] {stack_bb:g}bb push/fold solved: "
              f"SB jams {oracle.sb_jam_pct*100:.1f}% of hands, "
              f"BB calls {oracle.bb_call_pct*100:.1f}%  → {opath}", flush=True)
    return oracle


# ═══════════════════════════════════════════════════════════════════════════
# PRETTY-PRINT
# ═══════════════════════════════════════════════════════════════════════════

_GRID_RANKS = "AKQJT98765432"   # row/col headers, A→2


def format_grid(grid: np.ndarray, title: str) -> str:
    """13×13 frequency grid as a compact integer-percent block."""
    lines = [title, "     " + " ".join(f"{c:>3}" for c in _GRID_RANKS)]
    for r in range(13):
        cells = " ".join(f"{int(round(grid[r, c] * 100)):>3}" for c in range(13))
        lines.append(f"  {_GRID_RANKS[r]}  {cells}")
    return "\n".join(lines)
