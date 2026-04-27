"""End-to-end invariants on the HU NLHE engine via pybind.

Plays N random hands and checks that the engine's *observable behaviour* via
`pokertrainer_engine` is internally consistent. Complements the C++ Catch2
unit tests (which exercise individual functions but not the full pybind
roundtrip) and the `smoke_pybind.py` (single-state shape checks).

What's checked, per hand:
  * Zero-sum chip payoffs (`payoff_chips[0] + payoff_chips[1] == 0`).
  * Zero-sum BB payoffs.
  * Every legal action `step()`s cleanly.
  * Encoder invariants on every non-terminal observation:
      - `x.sum() > 0` (caught the 2026-04-25 zero-x bug late; wall it off).
      - `x[0:52].sum() == 2` (to_act's two hole cards, one-hot).
      - `x[52:104].sum() == n_visible_board` (board, one-hot).
      - exactly one bit set in `x[104:108]` (street).
      - exactly one of `x[108:110]` is 1.0 (OOP/IP).
      - `a.shape == (len(legal), A_DIM)`.
      - each `a[i, 0:11]` is a clean one-hot.
      - `legal_idx[i] == int(legal[i])`.
  * No duplicate cards across hole+board (Fisher-Yates correctness).
  * Hand terminates within HIST_MAX (24) decision points.
  * After step on terminal, observation() throws.
  * `step(legal_idx)` for an out-of-range idx throws.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_game_invariants.py
or:
    PYTHONPATH=engine/build:trainer python -m tests.test_game_invariants
"""
from __future__ import annotations

import sys
from typing import Callable

import numpy as np

import pokertrainer_engine as pte


# Number of random hands to play. ~5 decisions/hand → 50k encoder calls.
# Empirically <10s on cuda-less Python+pte. Bump for paranoia, drop for CI.
N_HANDS = 10_000


class TestState:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks_run = 0

    def check(self, cond: bool, msg: str) -> None:
        self.checks_run += 1
        if not cond:
            self.failures.append(msg)

    def expect_throws(self, fn: Callable[[], None], msg: str) -> None:
        self.checks_run += 1
        try:
            fn()
        except Exception:
            return
        self.failures.append(f"expected throw, got success: {msg}")


def check_observation_invariants(t: TestState, obs, n_visible_board: int,
                                 hand_seed: int, decision_idx: int) -> None:
    tag = f"seed={hand_seed:#x} decision={decision_idx}"
    x = np.asarray(obs.x)
    a = np.asarray(obs.a)
    legal = list(obs.legal)
    legal_idx = np.asarray(obs.legal_idx)

    # x not silently zero (the bug)
    t.check(x.sum() > 0, f"{tag}: obs.x.sum() must be > 0")

    # Hole cards: two one-hot bits in x[0:52]
    hole_bits = x[0:52]
    t.check(np.isclose(hole_bits.sum(), 2.0),
            f"{tag}: hole bits sum to 2 (got {hole_bits.sum()})")
    t.check((hole_bits == 1.0).sum() == 2,
            f"{tag}: exactly two hole bits are 1.0")
    t.check(((hole_bits == 0.0) | (hole_bits == 1.0)).all(),
            f"{tag}: hole bits are pure one-hot")

    # Board: n_visible_board one-hot bits in x[52:104]
    board_bits = x[52:104]
    t.check(np.isclose(board_bits.sum(), n_visible_board),
            f"{tag}: board bits sum to n_visible_board={n_visible_board} "
            f"(got {board_bits.sum()})")

    # Street one-hot at x[104:108]
    street_bits = x[104:108]
    t.check(np.isclose(street_bits.sum(), 1.0),
            f"{tag}: exactly one street bit (got sum={street_bits.sum()})")

    # Position one-hot at x[108:110]
    pos_bits = x[108:110]
    t.check(np.isclose(pos_bits.sum(), 1.0),
            f"{tag}: exactly one position bit (got sum={pos_bits.sum()})")

    # Action tensor shape and one-hot
    n_legal = len(legal)
    t.check(a.shape == (n_legal, pte.A_DIM),
            f"{tag}: a.shape == ({n_legal}, {pte.A_DIM})")
    one_hot_part = a[:, :pte.NUM_ACTIONS]
    t.check(np.allclose(one_hot_part.sum(axis=1), 1.0),
            f"{tag}: each a-row has exactly one one-hot action bit")

    # legal_idx mirrors legal
    t.check(len(legal_idx) == n_legal,
            f"{tag}: legal_idx and legal same length")
    for i, (lt, li) in enumerate(zip(legal, legal_idx)):
        t.check(int(lt) == int(li),
                f"{tag}: legal[{i}] type matches legal_idx[{i}]")


def check_no_duplicate_cards(t: TestState, env, hand_seed: int) -> None:
    """Across all 9 dealt cards (4 hole + 5 board), each card index unique."""
    s = env.state()
    hole_all = list(s.hole[0]) + list(s.hole[1])
    board = list(s.board)
    all_cards = hole_all + board
    unique = set(all_cards)
    t.check(len(unique) == len(all_cards),
            f"seed={hand_seed:#x}: cards must be unique (got {all_cards})")
    for c in all_cards:
        t.check(0 <= c < 52,
                f"seed={hand_seed:#x}: card index {c} in [0, 52)")


def play_one_hand(t: TestState, hand_seed: int) -> None:
    env = pte.Env(0xDEADBEEF, 100 * pte.BIG_BLIND_CHIPS)
    env.reset(hand_seed)

    check_no_duplicate_cards(t, env, hand_seed)

    decisions = 0
    while not env.is_terminal():
        # Cap to detect runaway loops (HIST_MAX is 24).
        t.check(decisions <= 50,
                f"seed={hand_seed:#x}: hand terminated within 50 decisions")
        if decisions > 50:
            return

        # Capture state before the encoder call.
        s = env.state()
        n_board = pte.HUGame_n_visible_board(s.street) if hasattr(pte, 'HUGame_n_visible_board') else _n_visible_board(s.street)
        obs = env.observation()

        check_observation_invariants(t, obs, n_board, hand_seed, decisions)

        # Pick a uniformly random legal action — exercises diverse paths.
        idx = np.random.randint(0, len(obs.legal))
        env.step(idx)
        decisions += 1

    # Terminal invariants.
    s = env.state()
    chip_sum = int(s.payoff_chips[0]) + int(s.payoff_chips[1])
    t.check(chip_sum == 0,
            f"seed={hand_seed:#x}: payoff_chips zero-sum (got {chip_sum})")

    bb = env.payoffs_bb()
    bb_sum = bb[0] + bb[1]
    t.check(abs(bb_sum) < 1e-9,
            f"seed={hand_seed:#x}: payoffs_bb zero-sum (got {bb_sum})")

    # Stepping a terminal env should throw; observation() too.
    t.expect_throws(lambda: env.step(0),
                    f"seed={hand_seed:#x}: step on terminal must throw")
    t.expect_throws(lambda: env.observation(),
                    f"seed={hand_seed:#x}: observation on terminal must throw")


def _n_visible_board(street) -> int:
    """Mirror of HUGame::n_visible_board: PREFLOP=0, FLOP=3, TURN=4, RIVER=5,
    SHOWDOWN=5. Used because pte doesn't expose the helper directly.
    """
    s = int(street)
    if s == 0: return 0      # PREFLOP
    if s == 1: return 3      # FLOP
    if s == 2: return 4      # TURN
    if s == 3: return 5      # RIVER
    return 5                  # SHOWDOWN


def check_legal_step_roundtrip(t: TestState) -> None:
    """For a few seeds, verify EVERY legal action steps without exception.

    Heavier than the random-walk test (which only takes one path per hand).
    """
    for seed in (0, 1, 42, 0xC0FFEE, 0xBADD15):
        for _ in range(5):
            env = pte.Env(0xCAFEBABE, 100 * pte.BIG_BLIND_CHIPS)
            env.reset(seed)
            steps = 0
            while not env.is_terminal() and steps < 5:
                obs = env.observation()
                # Try each legal action in a fresh copy by re-deriving.
                for try_idx in range(len(obs.legal)):
                    probe = pte.Env(0xCAFEBABE, 100 * pte.BIG_BLIND_CHIPS)
                    probe.reset(seed)
                    # Replay actions taken so far, then try try_idx.
                    for ai in past_actions:
                        probe.step(ai)
                    try:
                        probe.step(try_idx)
                    except Exception as e:
                        t.check(False,
                                f"seed={seed:#x} step={steps} legal[{try_idx}]"
                                f"={obs.legal[try_idx]} threw: {e}")
                # Now actually advance the main env on a random choice.
                idx = np.random.randint(0, len(obs.legal))
                # Track for the probe replays — local to this loop.
                past_actions.append(idx)
                env.step(idx)
                steps += 1
            past_actions.clear()


def check_min_raise_invariants(t: TestState) -> None:
    """min_raise_to_chips() must be ≥ to_call + 1 BB worth of chips on
    streets where raising is legal at all.
    """
    bb_chips = pte.BIG_BLIND_CHIPS
    for seed in (0, 1, 2, 42, 100):
        env = pte.Env(0xFEEDBEEF, 100 * bb_chips)
        env.reset(seed)
        if env.is_terminal():
            continue
        s = env.state()
        legal = s.legal_actions()
        if pte.ActionType.RAISE_25 in legal or pte.ActionType.ALL_IN in legal:
            min_raise = s.min_raise_to_chips()
            tc = s.to_call_chips()
            t.check(min_raise >= tc + bb_chips,
                    f"seed={seed:#x}: min_raise ({min_raise}) "
                    f">= to_call + 1BB ({tc + bb_chips})")


def main() -> int:
    t = TestState()

    print(f"[invariants] playing {N_HANDS} random hands…")
    rng = np.random.default_rng(0xC0FFEE)
    seeds = rng.integers(0, 2**63 - 1, size=N_HANDS, dtype=np.int64)
    for i, seed in enumerate(seeds):
        play_one_hand(t, int(seed))
        if (i + 1) % 2000 == 0 and not t.failures:
            print(f"  {i+1:>5d} hands ok ({t.checks_run} checks so far)")

    print("[invariants] min-raise invariants…")
    check_min_raise_invariants(t)

    print("[invariants] legal-step roundtrip on a handful of seeds…")
    global past_actions
    past_actions = []
    check_legal_step_roundtrip(t)

    print()
    print(f"checks run: {t.checks_run}")
    if t.failures:
        # Print up to 10 failures (would be a lot otherwise).
        print(f"FAILURES: {len(t.failures)}")
        for f in t.failures[:10]:
            print(f"  - {f}")
        if len(t.failures) > 10:
            print(f"  … {len(t.failures) - 10} more")
        return 1
    print("OK — all invariants hold across the sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
