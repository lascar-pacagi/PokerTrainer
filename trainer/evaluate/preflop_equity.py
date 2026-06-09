"""Preflop all-in equity between the 169 canonical starting-hand classes.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS PROVIDES
═══════════════════════════════════════════════════════════════════════════════

The push/fold Nash solver (pushfold_solver.py) needs, for any pair of starting
hands, the all-in showdown equity (probability the first hand beats the second
over a random 5-card runout). We expose that as two 169×169 matrices:

    E[i][j]  — mean equity of class i vs class j  (P(i wins) + 0.5·P(tie))
    C[i][j]  — number of Monte-Carlo deals that landed (SB=i, BB=j)

Both are estimated together by dealing uniformly-random, fully-distinct hands
(2 + 2 hole cards + 5 board) and tallying the showdown. Uniform dealing makes
C[i][j] the *combo-weighted joint frequency* of the two classes — exactly the
weighting the solver wants when it sums a hand's EV over an opposing range, and
it gets card-removal (blocker) effects for free.

═══════════════════════════════════════════════════════════════════════════════
THE 169-CLASS INDEXING
═══════════════════════════════════════════════════════════════════════════════

Engine card encoding (engine/src/card.h): card = rank<<2 | suit, rank 0..12
(12 = Ace), suit 0..3. A 2-card hand collapses to one of 169 strategic classes:
13 pocket pairs, 78 suited, 78 offsuit.

`CLASS_LABELS[id]` is the poker shorthand ("AA", "AKs", "72o", …). The canonical
13×13 grid (pairs on the diagonal, suited upper-right, offsuit lower-left) is
produced by `grid_cell_class(row, col)`; rows/cols run A→2.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import pokertrainer_engine as pte


N_CLASSES = 169
_RANK_CHARS = "23456789TJQKA"   # index 0..12 → rank label (12 = Ace)


# ─── Canonical class enumeration: pairs, then suited, then offsuit ───────────

def _enumerate_classes() -> list[tuple[int, int, str]]:
    classes: list[tuple[int, int, str]] = []
    for r in range(12, -1, -1):                 # AA, KK, … 22
        classes.append((r, r, "p"))
    for hi in range(12, -1, -1):                # suited, high rank first
        for lo in range(hi - 1, -1, -1):
            classes.append((hi, lo, "s"))
    for hi in range(12, -1, -1):                # offsuit
        for lo in range(hi - 1, -1, -1):
            classes.append((hi, lo, "o"))
    return classes


_CLASSES = _enumerate_classes()
assert len(_CLASSES) == N_CLASSES, len(_CLASSES)

_INDEX = {key: i for i, key in enumerate(_CLASSES)}


def _label(hi: int, lo: int, kind: str) -> str:
    if kind == "p":
        return _RANK_CHARS[hi] * 2
    return _RANK_CHARS[hi] + _RANK_CHARS[lo] + kind


CLASS_LABELS: list[str] = [_label(hi, lo, kind) for (hi, lo, kind) in _CLASSES]
LABEL_TO_ID: dict[str, int] = {lab: i for i, lab in enumerate(CLASS_LABELS)}

# Combos per class (reference; the MC counts are what the solver actually uses).
COMBOS = np.array([6 if k == "p" else (4 if k == "s" else 12)
                   for (_, _, k) in _CLASSES], dtype=np.int64)


def class_of(card0: int, card1: int) -> int:
    """Map a 2-card hole hand (engine card ints) to its 0..168 class id."""
    r0, s0 = card0 >> 2, card0 & 3
    r1, s1 = card1 >> 2, card1 & 3
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    if hi == lo:
        kind = "p"
    elif s0 == s1:
        kind = "s"
    else:
        kind = "o"
    return _INDEX[(hi, lo, kind)]


# 52×52 lookup so the MC inner loop avoids recomputing class_of (self-pairs,
# i.e. same card twice, are never dealt, so the diagonal is left as -1).
_CLASS_LUT = np.full((52, 52), -1, dtype=np.int16)
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            _CLASS_LUT[_a, _b] = class_of(_a, _b)


def grid_cell_class(row: int, col: int) -> int:
    """Class id at (row, col) of the standard 13×13 grid; rows/cols run A→2.

    Diagonal = pairs, upper-right (col>row) = suited, lower-left = offsuit.
    """
    ri, rj = 12 - row, 12 - col          # rank at this row / column
    if row == col:
        return _INDEX[(ri, ri, "p")]
    if col > row:                         # ri > rj  → suited high=ri low=rj
        return _INDEX[(ri, rj, "s")]
    return _INDEX[(rj, ri, "o")]          # row>col → rj > ri → offsuit


# ═══════════════════════════════════════════════════════════════════════════
# MONTE-CARLO EQUITY MATRIX
# ═══════════════════════════════════════════════════════════════════════════


def build_equity_matrix(evaluator: "pte.HandEvaluator",
                        n_deals: int,
                        seed: int = 0xE119,
                        batch: int = 100_000,
                        progress: bool = False
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Estimate (E, C) over `n_deals` uniform-random distinct deals.

    E[i][j] = mean SB-equity of class i vs class j (P(win)+0.5 P(tie)).
    C[i][j] = number of deals observed with (SB class i, BB class j).

    Cells with C==0 (impossible combos given removal, or just unsampled) have
    E==0; callers must guard with C>0. Returns float64 E and int64 C.
    """
    rng = np.random.default_rng(seed)
    eq_sum = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    counts = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    lut = _CLASS_LUT
    ev7 = evaluator.evaluate7

    done = 0
    while done < n_deals:
        b = min(batch, n_deals - done)
        # Vectorised distinct-card sampling: argsort of random keys yields a
        # per-row permutation of the 52-card deck; take the first 9 columns.
        deals = np.argsort(rng.random((b, 52)), axis=1)[:, :9].astype(np.int64)
        for d in deals:
            sb0, sb1, bb0, bb1, c0, c1, c2, c3, c4 = (int(x) for x in d)
            r_sb = ev7([sb0, sb1, c0, c1, c2, c3, c4])
            r_bb = ev7([bb0, bb1, c0, c1, c2, c3, c4])
            i = lut[sb0, sb1]
            j = lut[bb0, bb1]
            # lower rank = stronger hand (engine convention).
            if r_sb < r_bb:
                eq_sum[i, j] += 1.0
            elif r_sb == r_bb:
                eq_sum[i, j] += 0.5
            counts[i, j] += 1
        done += b
        if progress:
            print(f"    [equity] {done:,}/{n_deals:,} deals", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        E = np.where(counts > 0, eq_sum / counts, 0.0)
    return E, counts


# ─── Disk cache ──────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, n_deals: int, seed: int) -> Path:
    return Path(cache_dir) / f"preflop_equity_n{n_deals}_s{seed:08x}.npz"


def load_or_build_equity(n_deals: int,
                         seed: int = 0xE119,
                         cache_dir: str | None = None,
                         evaluator: "pte.HandEvaluator | None" = None,
                         progress: bool = True
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Return (E, C), loading the cached .npz if present else building+caching."""
    cache_dir = cache_dir or os.environ.get("PT_ORACLE_CACHE", "runs/oracle_cache")
    path = _cache_path(cache_dir, n_deals, seed)
    if path.exists():
        with np.load(path) as z:
            return z["E"], z["C"]
    if progress:
        print(f"  [equity] building {n_deals:,}-deal equity matrix "
              f"(seed=0x{seed:x}) → {path}", flush=True)
    if evaluator is None:
        evaluator = pte.HandEvaluator.load_or_generate("")
    E, C = build_equity_matrix(evaluator, n_deals, seed, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, E=E, C=C)
    return E, C
