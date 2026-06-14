"""Public Belief State (PBS) and its value-net query encoding.
ReBeL idea → the conversion trick: the PBS is the "state", and encode_query is
what the value net sees · doc §2.

A PBS = (public state, both players' beliefs). The *public state* is everything
both players can observe — board, street, pot, who is to act, the amount to
call, the acting player's stack — and is hole-independent. The *beliefs* are the
two players' ranges over the 1326 hole combos.

``encode_query`` packs a PBS (as seen by a given traverser) into the flat float
vector the value net consumes. It mirrors ``write_query_to`` in the reference
``subgame_solving.cc``:

    [ player_id, traverser, <public scalars>, board multi-hot(52),
      reaches_p0 (normalized), reaches_p1 (normalized) ]

Two faithfulness points carried over from the reference:
  * the beliefs written into the query are **normalized to sum 1** (with the
    same ``kReachSmoothingEps`` so an all-zero reach maps to uniform, not NaN);
  * the net therefore learns values *per unit opponent mass* — the solver
    multiplies the net output by the opponent's true reach mass at the leaf.

The public scalars are normalized to keep the net input well-scaled: pot, the
amount-to-call and the acting stack are divided by ``POT_SCALE`` bb, and street
by 3 (river). ``query_dim(...)`` returns the resulting vector length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hand_index import NUM_HANDS, NUM_CARDS

# Reach smoothing eps (matches kReachSmoothingEps in the reference).
REACH_EPS = 1e-80
# Fixed scale (bb) for normalizing chip-amount scalars into the net input.
POT_SCALE = 400.0
N_PUBLIC_SCALARS = 5  # player_id, traverser, street, pot, to_call (+ stack below)
# scalars actually written: player_id, traverser, street, pot, to_call, stack
_N_SCALARS = 6


@dataclass
class PublicState:
    """Hole-independent public information at a node (the PBS minus beliefs)."""
    street: int                 # 0=preflop 1=flop 2=turn 3=river
    board: list[int]            # 0..5 cards (engine encoding rank<<2|suit)
    to_act: int                 # 0 (SB) or 1 (BB) to act
    pot_bb: float               # chips in the pot, in big blinds
    to_call_bb: float = 0.0     # amount the to-act player must call (bb)
    stack_bb: float = 0.0       # acting player's remaining stack (bb)


def query_dim() -> int:
    return _N_SCALARS + NUM_CARDS + 2 * NUM_HANDS


def _normalize_safe(reach: np.ndarray) -> np.ndarray:
    r = np.asarray(reach, dtype=np.float64) + REACH_EPS
    return r / r.sum()


def encode_query(traverser: int, ps: PublicState,
                 reach0: np.ndarray, reach1: np.ndarray) -> np.ndarray:
    """Pack a PBS (as seen by ``traverser``) into the net input vector."""
    q = np.empty(query_dim(), dtype=np.float64)
    i = 0
    q[i] = float(ps.to_act); i += 1
    q[i] = float(traverser); i += 1
    q[i] = ps.street / 3.0; i += 1
    q[i] = ps.pot_bb / POT_SCALE; i += 1
    q[i] = ps.to_call_bb / POT_SCALE; i += 1
    q[i] = ps.stack_bb / POT_SCALE; i += 1
    board = np.zeros(NUM_CARDS, dtype=np.float64)
    for c in ps.board:
        board[int(c)] = 1.0
    q[i:i + NUM_CARDS] = board; i += NUM_CARDS
    q[i:i + NUM_HANDS] = _normalize_safe(reach0); i += NUM_HANDS
    q[i:i + NUM_HANDS] = _normalize_safe(reach1); i += NUM_HANDS
    return q


def decode_query(q: np.ndarray):
    """Inverse of `encode_query` (used for oracle/value-fn tests and debugging).

    Returns ``(traverser, PublicState, reach0_norm, reach1_norm)``. The reaches
    come back normalized (the encoding is lossy on reach magnitude — that mass
    lives in the solver's leaf scaler, not the query)."""
    i = 0
    to_act = int(round(q[i])); i += 1
    traverser = int(round(q[i])); i += 1
    street = int(round(q[i] * 3.0)); i += 1
    pot_bb = float(q[i] * POT_SCALE); i += 1
    to_call_bb = float(q[i] * POT_SCALE); i += 1
    stack_bb = float(q[i] * POT_SCALE); i += 1
    board = [int(c) for c in np.flatnonzero(q[i:i + NUM_CARDS] > 0.5)]; i += NUM_CARDS
    reach0 = np.array(q[i:i + NUM_HANDS], dtype=np.float64); i += NUM_HANDS
    reach1 = np.array(q[i:i + NUM_HANDS], dtype=np.float64); i += NUM_HANDS
    ps = PublicState(street=street, board=board, to_act=to_act,
                     pot_bb=pot_bb, to_call_bb=to_call_bb, stack_bb=stack_bb)
    return traverser, ps, reach0, reach1
