"""Per-hand terminal values at the leaves of a public subgame.

A leaf value is, for each of the 1326 hero hands h, the expected payoff against
the opponent's *reach* over their 1326 hands (card-removal masked). In CFR the
reach carries the weighting, so these are unnormalized counterfactual values:

    showdown:  v[h] = stake * sum_j reach_opp[j] * sign(h beats j)   (j != h, board-free)
    fold:      v[h] = (+/-matched) * sum_j reach_opp[j]              (hand-independent)

Showdown uses the engine's 7-card evaluator once per hand (1326 calls per board)
to get a rank vector (lower rank = stronger), then a vectorized pairwise compare.
The engine is imported lazily so the rest of the package needs no .so to import.
"""
from __future__ import annotations

import numpy as np

from .hand_index import HAND_CARDS, CONFLICT, NUM_HANDS, board_free_mask

_EVALUATOR = None


def _evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        import pokertrainer_engine as pte
        _EVALUATOR = pte.HandEvaluator.load_or_generate("")
    return _EVALUATOR


def hand_ranks(board: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """7-card rank per hand on `board` (lower = stronger), plus a validity mask.

    Hands that conflict with the board are invalid (rank set to a large sentinel;
    they never carry reach, but the mask makes that explicit). Requires a full
    5-card board.
    """
    assert len(board) == 5, "hand_ranks needs a full 5-card board"
    ev = _evaluator()
    valid = board_free_mask(board)
    ranks = np.full(NUM_HANDS, np.iinfo(np.int32).max, dtype=np.int64)
    b = list(board)
    for h in range(NUM_HANDS):
        if valid[h]:
            c0, c1 = int(HAND_CARDS[h, 0]), int(HAND_CARDS[h, 1])
            ranks[h] = ev.evaluate7([c0, c1, b[0], b[1], b[2], b[3], b[4]])
    return ranks, valid


def showdown_sign_matrix(ranks: np.ndarray, valid: np.ndarray,
                         board: list[int]) -> np.ndarray:
    """(1326, 1326) float: +1 if hero hand i beats opp hand j, -1 if loses, 0
    tie/invalid. Masked so conflicting or board-using pairs contribute nothing.
    """
    # sign(rank_j - rank_i): hero wins when its rank is lower (stronger).
    diff = ranks[None, :].astype(np.int64) - ranks[:, None].astype(np.int64)
    sign = np.sign(diff).astype(np.float64)
    free = board_free_mask(board)
    mask = (~CONFLICT).astype(np.float64) * free[:, None] * free[None, :]
    return sign * mask


def showdown_values(sign_matrix: np.ndarray, opp_reach: np.ndarray,
                    stake: float) -> np.ndarray:
    """v[h] = stake * sum_j sign(h,j) * opp_reach[j]."""
    return stake * (sign_matrix @ opp_reach)


def fold_values(opp_reach: np.ndarray, matched: float, hero_wins: bool,
                board: list[int] | None = None) -> np.ndarray:
    """Per-hero-hand value when the hand ends on a fold (hand-independent up to
    card removal): +/-matched times the opponent reach the hero hand doesn't
    block. `hero_wins` = the opponent is the folder.
    """
    sign = 1.0 if hero_wins else -1.0
    free = board_free_mask(board) if board else np.ones(NUM_HANDS)
    # mass of opponent reach compatible with each hero hand (card removal).
    compat = (~CONFLICT).astype(np.float64) * free[None, :]
    return sign * matched * (compat @ opp_reach)
