"""Per-hand terminal values at the leaves of a public subgame.
ReBeL idea → leaf payoffs; the rank/incidence primitives that scaled the river
chance node (RiverShowdown, FoldRemoval) · doc §8.

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


class RiverShowdown:
    """Card-removal-correct river showdown WITHOUT a dense (H, H) sign matrix.

        v[h] = stake · Σ_{h' valid, h'∩h=∅} oppr[h'] · sign(rank_h' − rank_h)
             = stake · (WIN[h] − LOSE[h])

    where WIN/LOSE are split by inclusion-exclusion over h's two cards:

        WIN[h]  = (weaker mass, all hands) − (weaker mass sharing c0)
                                          − (weaker mass sharing c1)

    (no "shares both cards" term: the only hand with both of h's cards is h
    itself, which is never an opponent nor strictly weaker than itself). Every
    term is a gather into a rank-sorted cumulative sum, so a 1326² dense matvec
    becomes a few O(H) cumsums — and 48 of these (one per river card) cost a few
    rank vectors instead of 48 × 14 MB dense matrices. Matches
    ``showdown_sign_matrix @ opp_reach`` exactly.
    """

    def __init__(self, ranks: np.ndarray, valid: np.ndarray):
        H = NUM_HANDS
        r = np.asarray(ranks, dtype=np.int64)
        self.valid = np.asarray(valid, dtype=np.float64)
        self.c0 = HAND_CARDS[:, 0].astype(np.int64)
        self.c1 = HAND_CARDS[:, 1].astype(np.int64)

        # global rank order (ascending = strongest first)
        self.order = np.argsort(r, kind="stable")
        rs = r[self.order]
        self.lo_all = np.searchsorted(rs, r, side="left")   # #hands rank < r[h]
        self.hi_all = np.searchsorted(rs, r, side="right")  # #hands rank ≤ r[h]

        # per-card rank-sorted hand lists (only valid hands)
        from .hand_index import NUM_CARDS
        lists: list[list[int]] = [[] for _ in range(NUM_CARDS)]
        for h in np.flatnonzero(self.valid):
            lists[self.c0[h]].append(int(h)); lists[self.c1[h]].append(int(h))
        M = max((len(x) for x in lists), default=0)
        self.M = M
        self.card_hands = np.zeros((NUM_CARDS, M), dtype=np.int64)
        self.card_mask = np.zeros((NUM_CARDS, M), dtype=np.float64)
        card_ranks = np.full((NUM_CARDS, M), np.iinfo(np.int64).max, dtype=np.int64)
        for c in range(NUM_CARDS):
            hs = sorted(lists[c], key=lambda h: r[h])
            n = len(hs)
            if n:
                self.card_hands[c, :n] = hs
                self.card_mask[c, :n] = 1.0
                card_ranks[c, :n] = r[hs]
        # per-hand position within each of its two cards' sorted lists
        self.lo_c0 = np.zeros(H, dtype=np.int64); self.hi_c0 = np.zeros(H, dtype=np.int64)
        self.lo_c1 = np.zeros(H, dtype=np.int64); self.hi_c1 = np.zeros(H, dtype=np.int64)
        for h in np.flatnonzero(self.valid):
            cr0 = card_ranks[self.c0[h]]; cr1 = card_ranks[self.c1[h]]
            self.lo_c0[h] = np.searchsorted(cr0, r[h], side="left")
            self.hi_c0[h] = np.searchsorted(cr0, r[h], side="right")
            self.lo_c1[h] = np.searchsorted(cr1, r[h], side="left")
            self.hi_c1[h] = np.searchsorted(cr1, r[h], side="right")

    def values(self, opp_reach: np.ndarray, stake: float) -> np.ndarray:
        ov = np.asarray(opp_reach, dtype=np.float64) * self.valid
        cum = np.empty(NUM_HANDS + 1)
        cum[0] = 0.0
        np.cumsum(ov[self.order], out=cum[1:])
        total = cum[-1]
        stronger_all = cum[self.lo_all]
        weaker_all = total - cum[self.hi_all]

        occ = ov[self.card_hands] * self.card_mask          # (NUM_CARDS, M)
        cumc = np.zeros((occ.shape[0], self.M + 1))
        np.cumsum(occ, axis=1, out=cumc[:, 1:])
        total_c = cumc[:, -1]
        stronger_c0 = cumc[self.c0, self.lo_c0]
        weaker_c0 = total_c[self.c0] - cumc[self.c0, self.hi_c0]
        stronger_c1 = cumc[self.c1, self.lo_c1]
        weaker_c1 = total_c[self.c1] - cumc[self.c1, self.hi_c1]

        win = weaker_all - weaker_c0 - weaker_c1
        lose = stronger_all - stronger_c0 - stronger_c1
        return stake * (win - lose) * self.valid


class FoldRemoval:
    """Fold value without a dense (H, H) matrix.

        v[h] = sign · matched · Σ_{h' valid, h'∩h=∅} opp_reach[h']

    The card-removal sum is inclusion-exclusion over h's two cards:

        Σ_{h'∩h=∅} = total − S[c0] − S[c1] + opp_reach[h]

    where S[c] = opp mass on hands containing card c (the +opp_reach[h] adds back
    the only hand counted in both S[c0] and S[c1] — h itself — which the sum must
    exclude). The card→hand incidence is board-independent, so a single shared
    (52, H) matrix serves every river card; only the validity mask is per-board.
    O(52·H) and ~70 KB shared, vs a 14 MB dense matrix per fold terminal.
    """

    _INCIDENCE = None  # (NUM_CARDS, H) 0/1, shared

    def __init__(self, valid: np.ndarray):
        from .hand_index import NUM_CARDS
        self.valid = np.asarray(valid, dtype=np.float64)
        self.c0 = HAND_CARDS[:, 0].astype(np.int64)
        self.c1 = HAND_CARDS[:, 1].astype(np.int64)
        if FoldRemoval._INCIDENCE is None:
            inc = np.zeros((NUM_CARDS, NUM_HANDS))
            idx = np.arange(NUM_HANDS)
            inc[HAND_CARDS[:, 0], idx] = 1.0
            inc[HAND_CARDS[:, 1], idx] = 1.0
            FoldRemoval._INCIDENCE = inc

    def values(self, opp_reach: np.ndarray, sign: float, matched: float) -> np.ndarray:
        ov = np.asarray(opp_reach, dtype=np.float64) * self.valid   # mask invalid OPP
        total = ov.sum()
        S = FoldRemoval._INCIDENCE @ ov                     # (NUM_CARDS,)
        # Σ_{h'∩h=∅}, for EVERY hero hand. We deliberately do NOT mask invalid
        # hero hands: the dense `make_fold_value` (compat @ opp) leaves them
        # nonzero too (they carry zero reach, so harmless), and the incidence
        # formula already reproduces that value — an invalid h=(board_card, x)
        # has S[board_card]=0, so kept = total − S[x], exactly the dense row.
        # Masking hero here would make this differ from the dense fold.
        kept = total - S[self.c0] - S[self.c1] + ov
        return sign * matched * kept


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
