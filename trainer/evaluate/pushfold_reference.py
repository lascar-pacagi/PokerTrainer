"""Published HU Nash push/fold ranges — a sanity check on our solved oracle.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════════════════

pushfold_solver.py computes the equilibrium from our engine's own equities
("oracle A"). This module is the independent cross-check the user asked for:
parse a *published* Nash range string into the 169-class space and compare it to
the solved oracle. If our solver is correct, the two should agree on all but a
few boundary hands.

CAVEAT — published charts disagree at the edges. Different sources use chip-EV
vs ICM, antes vs none, and slightly different effective stacks, so a perfect
169/169 match is neither expected nor meaningful. We score combo-weighted
decision agreement and list the disagreements; ~90%+ agreement with the right
*shape* (monotone strength frontier) is the bar.

The 10 bb references below are chip-EV, no-ante, blind-vs-blind:

  SB open-shove  ≈ 57.5% of hands   (gamblingcalc.com/poker/push-fold-chart)
  BB call-vs-jam ≈ 42.7% (at ~9 bb) (pokerstrategy / HoldemResources HUNE)

The SB string expands to ~57.5% of combos, matching our solver's ~57–59%; the
percentage some pages print alongside it is often mislabeled, so we trust the
range *string*, not the quoted percent.
"""
from __future__ import annotations

import numpy as np

from .preflop_equity import N_CLASSES, CLASS_LABELS, LABEL_TO_ID, COMBOS

_RANKS = "23456789TJQKA"          # index 0..12
_RIDX = {c: i for i, c in enumerate(_RANKS)}


# ═══════════════════════════════════════════════════════════════════════════
# RANGE-STRING PARSER
# ═══════════════════════════════════════════════════════════════════════════
#
# Supports the standard shorthand:
#   "22+"          pairs from 22 up to AA
#   "A2s+"/"K4o+"  same high card, low card walking up to (high-1), fixed suit
#   "Q4s"/"T9o"    a single suited/offsuit combo class
#   "AKs"          single class
# Tokens are comma-separated; whitespace is ignored.


def _expand_token(tok: str) -> list[str]:
    tok = tok.strip()
    plus = tok.endswith("+")
    if plus:
        tok = tok[:-1]
    hi, lo = _RIDX[tok[0]], _RIDX[tok[1]]
    if hi == lo:                                   # pocket pair
        if not plus:
            return [tok]
        return [_RANKS[r] * 2 for r in range(lo, 13)]
    suit = tok[2]                                  # 's' or 'o'
    hi, lo = max(hi, lo), min(hi, lo)
    if not plus:
        return [f"{_RANKS[hi]}{_RANKS[lo]}{suit}"]
    # "+" on a non-pair walks the LOW card up toward (hi-1).
    return [f"{_RANKS[hi]}{_RANKS[r]}{suit}" for r in range(lo, hi)]


def parse_range(s: str) -> frozenset[int]:
    """Parse a poker range string into a set of 0..168 class ids."""
    ids: set[int] = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        for lab in _expand_token(tok):
            ids.add(LABEL_TO_ID[lab])
    return frozenset(ids)


# ═══════════════════════════════════════════════════════════════════════════
# PUBLISHED 10 bb REFERENCES
# ═══════════════════════════════════════════════════════════════════════════

SB_SHOVE_10BB = ("22+, A2s+, A2o+, K2s+, K2o+, Q2s+, Q3o+, J2s+, J7o+, "
                 "T6s+, T8o+, 97s+, 98o")
BB_CALL_10BB = ("22+, A2s+, A2o+, K2s+, K4o+, Q4s+, Q8o+, J7s+, J9o+, "
                "T8s+, T9o, 98s")


def reference_pct(range_string: str) -> float:
    ids = parse_range(range_string)
    mask = np.zeros(N_CLASSES, dtype=bool)
    for i in ids:
        mask[i] = True
    return float(COMBOS[mask].sum() / COMBOS.sum())


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════


def compare(freq: np.ndarray, ref: frozenset[int], *, threshold: float = 0.5
            ) -> dict:
    """Compare a solved frequency vector to a published reference hand-set.

    Returns combo-weighted decision agreement plus the boundary disagreements:
      only_solved : hands the oracle plays (freq≥thr) but the reference doesn't
      only_ref    : hands the reference plays but the oracle doesn't
    """
    in_ref = np.zeros(N_CLASSES, dtype=bool)
    for i in ref:
        in_ref[i] = True
    plays = freq >= threshold
    w = COMBOS.astype(np.float64)
    agree = float((w * (plays == in_ref)).sum() / w.sum())
    only_solved = [CLASS_LABELS[i] for i in range(N_CLASSES) if plays[i] and not in_ref[i]]
    only_ref    = [CLASS_LABELS[i] for i in range(N_CLASSES) if in_ref[i] and not plays[i]]
    return {
        "agreement": agree,
        "solved_pct": float((w * plays).sum() / w.sum()),
        "ref_pct": float(w[in_ref].sum() / w.sum()),
        "only_solved": only_solved,
        "only_ref": only_ref,
    }


def format_comparison(oracle, *, threshold: float = 0.5) -> str:
    """Human-readable SB-jam and BB-call comparison vs the published ranges."""
    sb = compare(oracle.sb_jam, parse_range(SB_SHOVE_10BB), threshold=threshold)
    bb = compare(oracle.bb_call, parse_range(BB_CALL_10BB), threshold=threshold)
    lines = [f"[pushfold-ref] solved oracle vs published {oracle.stack_bb:g}bb Nash:"]
    for name, d in (("SB-jam ", sb), ("BB-call", bb)):
        lines.append(f"  {name}: agree={d['agreement']*100:5.1f}%  "
                     f"solved={d['solved_pct']*100:4.1f}%  ref={d['ref_pct']*100:4.1f}%")
        if d["only_solved"]:
            lines.append(f"           solved-only: {' '.join(d['only_solved'])}")
        if d["only_ref"]:
            lines.append(f"           ref-only   : {' '.join(d['only_ref'])}")
    return "\n".join(lines)
