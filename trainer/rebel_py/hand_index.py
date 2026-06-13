"""The 1326 hole-card combos: enumeration, indexing, and card-removal masks.

A "hand" in poker ReBeL is a specific 2-card combo (not the 169 strategic
classes — that abstraction is for the push/fold oracle). There are C(52,2)=1326
combos. Beliefs, regrets and strategies in the subgame solver are vectors/
matrices indexed by these 1326 hands.

Card encoding matches the engine (engine/src/card.h): ``card = rank<<2 | suit``,
rank 0..12 = 2..A, suit 0..3 = c d h s, so a card is an int in 0..51.

Everything here is pure numpy + plain ints so the solver can stay vectorized and
the module imports without the engine (the engine is only needed for showdown
ranks, handled in showdown.py).
"""
from __future__ import annotations

import numpy as np

NUM_CARDS = 52
NUM_HANDS = 1326  # C(52, 2)

_RANK_CHARS = "23456789TJQKA"
_SUIT_CHARS = "cdhs"


def card_str(card: int) -> str:
    return f"{_RANK_CHARS[card >> 2]}{_SUIT_CHARS[card & 3]}"


# ─── canonical combo enumeration (c0 < c1), fixed order ─────────────────────

def _enumerate() -> np.ndarray:
    out = np.empty((NUM_HANDS, 2), dtype=np.int64)
    k = 0
    for c0 in range(NUM_CARDS):
        for c1 in range(c0 + 1, NUM_CARDS):
            out[k, 0] = c0
            out[k, 1] = c1
            k += 1
    assert k == NUM_HANDS, k
    return out


# HAND_CARDS[i] = (c0, c1) for hand i, c0 < c1.
HAND_CARDS: np.ndarray = _enumerate()

# 52x52 lookup: index of the combo {a, b} (-1 on the diagonal).
_HAND_OF = np.full((NUM_CARDS, NUM_CARDS), -1, dtype=np.int64)
for _i in range(NUM_HANDS):
    _a, _b = int(HAND_CARDS[_i, 0]), int(HAND_CARDS[_i, 1])
    _HAND_OF[_a, _b] = _i
    _HAND_OF[_b, _a] = _i


def hand_index(card_a: int, card_b: int) -> int:
    """Index of the combo holding cards a and b (order-independent)."""
    return int(_HAND_OF[card_a, card_b])


# Bitmask (uint64) with the two card bits set, per hand — for fast conflicts.
HAND_BITS: np.ndarray = (
    (np.uint64(1) << HAND_CARDS[:, 0].astype(np.uint64))
    | (np.uint64(1) << HAND_CARDS[:, 1].astype(np.uint64))
)

# CONFLICT[i, j] = True iff hands i and j share a card (cannot be held at once).
CONFLICT: np.ndarray = (HAND_BITS[:, None] & HAND_BITS[None, :]) != np.uint64(0)


def board_bits(board: list[int]) -> np.uint64:
    m = np.uint64(0)
    for c in board:
        if 0 <= c < NUM_CARDS:
            m |= np.uint64(1) << np.uint64(c)
    return m


def board_free_mask(board: list[int]) -> np.ndarray:
    """Boolean (1326,): True for combos that use none of the board cards."""
    bb = board_bits(board)
    return (HAND_BITS & bb) == np.uint64(0)


def opponent_reach_matrix(board: list[int] | None = None) -> np.ndarray:
    """(1326, 1326) float: 1.0 where hands i and j can co-exist (no shared
    card, and — if a board is given — neither conflicts with the board), else 0.

    Used to mask an opponent belief vector per hero hand at showdown / for
    card-removal-correct reach products.
    """
    ok = (~CONFLICT).astype(np.float64)
    if board:
        free = board_free_mask(board)
        ok = ok * free[:, None] * free[None, :]
    return ok
