"""ctypes smoke test for the inspector C ABI.

Exercises the new symbols added for the Flutter UI:
  pt_env_state_snapshot, pt_env_history_at, pt_env_legal_sizings,
  pt_env_step_action, pt_env_set_hole / set_board / get_hole / get_board.

Pins the PtStateSnapshot / PtHistoryEntry layout: any drift in the C struct
will trip the ctypes mirror's expected sizes and surface immediately.

Run:
    PYTHONPATH=engine/build python engine/tests/smoke_ffi.py
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SO_PATH = ROOT / "engine" / "build" / "libpokertrainer.so"
lib = C.CDLL(str(SO_PATH))


# ─── struct mirrors (must match ffi.cpp layout exactly) ────────────────────
class PtStateSnapshot(C.Structure):
    _fields_ = [
        ("pot_chips",                 C.c_int64),
        ("stacks_chips",              C.c_int64 * 2),
        ("invested_this_street",      C.c_int64 * 2),
        ("invested_prior_streets",    C.c_int64 * 2),
        ("to_call_chips",             C.c_int64),
        ("min_raise_to_chips",        C.c_int64),
        ("payoff_chips",              C.c_int64 * 2),
        ("street",                    C.c_int32),
        ("to_act",                    C.c_int32),
        ("terminal",                  C.c_int32),
        ("is_terminal",               C.c_int32),
        ("all_in",                    C.c_int32 * 2),
        ("history_size",              C.c_int32),
        ("_padding",                  C.c_int32),
    ]


class PtHistoryEntry(C.Structure):
    _fields_ = [
        ("bet_to_chips",      C.c_int64),
        ("pot_after_chips",   C.c_int64),
        ("stack_after_chips", C.c_int64),
        ("actor",             C.c_int32),
        ("street",            C.c_int32),
        ("type",              C.c_int32),
        ("was_all_in",        C.c_int32),
    ]


# Lock down struct sizes — must agree with the C side's static_asserts.
assert C.sizeof(PtStateSnapshot) == 120, C.sizeof(PtStateSnapshot)
assert C.sizeof(PtHistoryEntry)  == 40,  C.sizeof(PtHistoryEntry)


# ─── prototypes ────────────────────────────────────────────────────────────
lib.pt_eval_init.argtypes = [C.c_char_p]
lib.pt_eval_init.restype  = C.c_int32

lib.pt_x_dim.restype       = C.c_int32
lib.pt_a_dim.restype       = C.c_int32
lib.pt_num_actions.restype = C.c_int32
lib.pt_hist_max.restype    = C.c_int32
lib.pt_hist_feat.restype   = C.c_int32

lib.pt_x_off_street.argtypes = [C.c_int32]
lib.pt_x_off_street.restype  = C.c_int32
lib.pt_street_slots.argtypes = [C.c_int32]
lib.pt_street_slots.restype  = C.c_int32

lib.pt_env_create.argtypes        = [C.c_uint64, C.c_int64]
lib.pt_env_create.restype         = C.c_void_p
lib.pt_env_destroy.argtypes       = [C.c_void_p]
lib.pt_env_destroy.restype        = None
lib.pt_env_reset_seed.argtypes    = [C.c_void_p, C.c_uint64]
lib.pt_env_reset_seed.restype     = C.c_int32
lib.pt_env_is_terminal.argtypes   = [C.c_void_p]
lib.pt_env_is_terminal.restype    = C.c_int32

lib.pt_env_state_snapshot.argtypes = [C.c_void_p, C.POINTER(PtStateSnapshot)]
lib.pt_env_state_snapshot.restype  = C.c_int32
lib.pt_env_history_at.argtypes     = [C.c_void_p, C.c_int32, C.POINTER(PtHistoryEntry)]
lib.pt_env_history_at.restype      = C.c_int32
lib.pt_env_legal_sizings.argtypes  = [C.c_void_p, C.POINTER(C.c_int64)]
lib.pt_env_legal_sizings.restype   = C.c_int32
lib.pt_env_step_action.argtypes    = [C.c_void_p, C.c_int32, C.POINTER(C.c_double)]
lib.pt_env_step_action.restype     = C.c_int32

lib.pt_env_clear_cards.argtypes = [C.c_void_p]
lib.pt_env_clear_cards.restype  = C.c_int32
lib.pt_env_set_hole.argtypes  = [C.c_void_p, C.c_int32, C.c_uint8, C.c_uint8]
lib.pt_env_set_hole.restype   = C.c_int32
lib.pt_env_set_board.argtypes = [C.c_void_p, C.POINTER(C.c_uint8)]
lib.pt_env_set_board.restype  = C.c_int32
lib.pt_env_get_hole.argtypes  = [C.c_void_p, C.c_int32, C.POINTER(C.c_uint8), C.POINTER(C.c_uint8)]
lib.pt_env_get_hole.restype   = C.c_int32
lib.pt_env_get_board.argtypes = [C.c_void_p, C.POINTER(C.c_uint8)]
lib.pt_env_get_board.restype  = C.c_int32

lib.pt_make_card.argtypes = [C.c_int32, C.c_int32]
lib.pt_make_card.restype  = C.c_uint8


def card(rank: int, suit: int) -> int:
    """rank in 0..12 (2..A), suit in 0..3 (cdhs)."""
    return lib.pt_make_card(rank, suit)


NO_CARD       = 0xFF
BIG_BLIND     = 100
SMALL_BLIND   = 50
ACTION_FOLD       = 0
ACTION_CHECK_CALL = 1
ACTION_RAISE_25   = 2
ACTION_ALL_IN     = 10


def main() -> None:
    assert lib.pt_eval_init(b"") == 0

    # Constants sanity (v0.4).
    assert lib.pt_x_dim()       == 812
    assert lib.pt_a_dim()       == 11
    assert lib.pt_num_actions() == 11
    assert lib.pt_hist_max()    == 34
    assert lib.pt_hist_feat()   == 20
    assert lib.pt_x_off_street(0) == 132
    assert lib.pt_street_slots(0) == 10
    assert lib.pt_street_slots(1) == 8
    assert lib.pt_street_slots(2) == 8
    assert lib.pt_street_slots(3) == 8

    env = lib.pt_env_create(0x2026_04_30, 100 * BIG_BLIND)
    assert env is not None

    # ─── 1. State snapshot at fresh deal ────────────────────────────────────
    snap = PtStateSnapshot()
    assert lib.pt_env_state_snapshot(env, C.byref(snap)) == 0
    assert snap.pot_chips             == SMALL_BLIND + BIG_BLIND  # 150
    assert snap.stacks_chips[0]       == 100 * BIG_BLIND - SMALL_BLIND  # SB posted 50
    assert snap.stacks_chips[1]       == 100 * BIG_BLIND - BIG_BLIND    # BB posted 100
    assert snap.invested_this_street[0] == SMALL_BLIND
    assert snap.invested_this_street[1] == BIG_BLIND
    assert snap.to_call_chips         == BIG_BLIND - SMALL_BLIND  # SB owes 50
    assert snap.street                == 0   # PREFLOP
    assert snap.to_act                == 0   # SB
    assert snap.is_terminal           == 0
    assert snap.history_size          == 0
    assert snap.all_in[0]             == 0
    assert snap.all_in[1]             == 0

    # ─── 2. legal_sizings sanity preflop ────────────────────────────────────
    sizings = (C.c_int64 * 11)()
    n = lib.pt_env_legal_sizings(env, sizings)
    assert n == 11
    # FOLD is always legal preflop when SB is facing the BB.
    assert sizings[ACTION_FOLD] == 0
    # CHECK_CALL means call 50 chips → total invest 100 chips (the BB amount).
    assert sizings[ACTION_CHECK_CALL] == BIG_BLIND
    # ALL_IN is shove the whole stack: SB posted 50, has 9950 left → total invest 10000 chips.
    assert sizings[ACTION_ALL_IN] == 100 * BIG_BLIND

    # ─── 3. Hole-card override + uniqueness validation ──────────────────────
    # Clear the random deal first so the explicit set_hole calls don't trip
    # against cards that happened to land on the board / opp's hole.
    assert lib.pt_env_clear_cards(env) == 0

    AsKs = (card(12, 3), card(11, 3))   # ace of spades, king of spades
    KhQh = (card(11, 2), card(10, 2))   # king of hearts, queen of hearts

    assert lib.pt_env_set_hole(env, 0, AsKs[0], AsKs[1]) == 0
    assert lib.pt_env_set_hole(env, 1, KhQh[0], KhQh[1]) == 0

    # Read-back round-trip.
    c0, c1 = C.c_uint8(), C.c_uint8()
    assert lib.pt_env_get_hole(env, 0, C.byref(c0), C.byref(c1)) == 0
    assert (c0.value, c1.value) == AsKs

    # Conflict: try to give BB the same king-of-spades SB already has.
    assert lib.pt_env_set_hole(env, 1, card(11, 3), card(9, 0)) == -1
    # Internal state should remain KhQh after a rejected set.
    assert lib.pt_env_get_hole(env, 1, C.byref(c0), C.byref(c1)) == 0
    assert (c0.value, c1.value) == KhQh

    # Same card twice in one pair is rejected.
    assert lib.pt_env_set_hole(env, 0, card(0, 0), card(0, 0)) == -1

    # ─── 4. Board override ──────────────────────────────────────────────────
    new_board = (C.c_uint8 * 5)(card(2, 0), card(7, 1), card(11, 0), NO_CARD, NO_CARD)
    assert lib.pt_env_set_board(env, new_board) == 0
    out_b = (C.c_uint8 * 5)()
    assert lib.pt_env_get_board(env, out_b) == 0
    assert tuple(out_b) == (card(2, 0), card(7, 1), card(11, 0), NO_CARD, NO_CARD)

    # Board conflicting with hole cards is rejected.
    bad_board = (C.c_uint8 * 5)(AsKs[0], card(7, 1), card(11, 0), NO_CARD, NO_CARD)
    assert lib.pt_env_set_board(env, bad_board) == -1
    # Should not have mutated.
    assert lib.pt_env_get_board(env, out_b) == 0
    assert tuple(out_b) == (card(2, 0), card(7, 1), card(11, 0), NO_CARD, NO_CARD)

    # ─── 5. step_action + history_at round-trip ─────────────────────────────
    reward = C.c_double(0.0)
    # SB raise to RAISE_100 (full pot) preflop.
    assert lib.pt_env_step_action(env, ACTION_RAISE_25, C.byref(reward)) == 0
    snap2 = PtStateSnapshot()
    lib.pt_env_state_snapshot(env, C.byref(snap2))
    assert snap2.history_size == 1
    assert snap2.to_act       == 1   # BB now to act
    assert snap2.pot_chips    > snap.pot_chips

    h0 = PtHistoryEntry()
    assert lib.pt_env_history_at(env, 0, C.byref(h0)) == 0
    assert h0.actor == 0      # SB acted
    assert h0.street == 0     # preflop
    assert h0.type == ACTION_RAISE_25
    assert h0.bet_to_chips > BIG_BLIND  # raise resolved to something > 100 chips
    assert h0.was_all_in == 0

    # Out-of-range history idx returns -1.
    assert lib.pt_env_history_at(env, 99, C.byref(h0)) == -1

    # ─── 6. Illegal step_action returns -1 ──────────────────────────────────
    # CHECK_CALL is BB's only sensible response besides fold/raise. Try FOLD then
    # confirm we can't step on a terminal env afterwards.
    lib.pt_env_step_action(env, ACTION_FOLD, C.byref(reward))   # BB folds → terminal
    assert lib.pt_env_is_terminal(env) == 1
    assert lib.pt_env_step_action(env, ACTION_FOLD, C.byref(reward)) == -1

    # Snapshot on terminal: payoffs filled, to_act = -1.
    snap3 = PtStateSnapshot()
    lib.pt_env_state_snapshot(env, C.byref(snap3))
    assert snap3.is_terminal == 1
    assert snap3.to_act      == -1
    assert snap3.terminal    == 1   # FOLD
    assert snap3.payoff_chips[0] + snap3.payoff_chips[1] == 0

    lib.pt_env_destroy(env)
    print("OK — inspector ABI smoke passed.")


if __name__ == "__main__":
    main()
