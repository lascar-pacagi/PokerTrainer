"""Match runner: play N hands between two policies with seat alternation.

Seat alternation is critical for HU win-rate measurement: SB has positional
equity postflop and initiative preflop, so playing N hands with fixed seats
biases the result. Two schemes are supported:

  Regular:    flip seats every hand. Cheap; variance dominated by card luck.
  Duplicated: every "pair" of hands replays the *same deck* with seats swapped.
              Card-luck variance cancels across the pair, leaving mostly
              position asymmetry + decision skill. Typically 2–5× smaller SE
              at the same hand count. Only works because `Env.reset(seed)` is
              fully deterministic on the deck — see engine/src/env.cpp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

import pokertrainer_engine as pte


NUM_ACTIONS = int(pte.NUM_ACTIONS)             # 11: FOLD..ALL_IN
NUM_STREETS = 4                                # preflop / flop / turn / river

# Short per-slot labels for compact printing. Order matches ActionType indices.
SLOT_LABELS = ("F", "C", "R25", "R33", "R50", "R75",
               "R100", "R150", "R200", "R300", "AI")
STREET_LABELS = ("pre", "flop", "turn", "rvr")
assert len(SLOT_LABELS) == NUM_ACTIONS
assert len(STREET_LABELS) == NUM_STREETS


@dataclass
class MatchResult:
    n_hands: int
    a_bb_total: float       # policy A's total chip delta in BB, across all hands and both seats
    b_bb_total: float
    a_name: str
    b_name: str
    elapsed_s: float
    # Per-street, per-slot counts of actions chosen BY policy A (not B).
    # Shape (NUM_STREETS, NUM_ACTIONS). Lets us answer "did the model ever pick
    # RAISE_25 postflop?" — the whole reason that slot exists (see
    # docs/STATE_ENCODING.md).
    a_slot_counts: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_STREETS, NUM_ACTIONS), dtype=np.int64))
    # Per-unit bb deltas for A. In regular mode each unit is one hand; in
    # duplicated mode each unit is a 2-hand pair (same deck, swapped seats).
    # Used for empirical SE — the "unit" is the correlation-preserving sample.
    a_unit_deltas_bb: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64))
    hands_per_unit: int = 1                     # 1 for regular, 2 for duplicated
    duplicated: bool = False

    @property
    def a_mbb_per_hand(self) -> float:
        return 1000.0 * self.a_bb_total / max(1, self.n_hands)

    @property
    def a_mbb_per_100(self) -> float:
        return 100.0 * self.a_mbb_per_hand

    @property
    def a_std_err_mbb_per_100(self) -> float:
        """Empirical SE on the per-100-hand win rate.

        For deltas `d[i]` over n_units independent units, each worth
        `hands_per_unit` hands:
          per-hand win rate = mean(d) / hands_per_unit
          SE(per-hand)      = SD(d) / (hands_per_unit * sqrt(n_units))
        Conversion to mbb/100: × 1000 (bb→mbb) × 100 (per-100) = × 100_000.
        """
        d = self.a_unit_deltas_bb
        if d.size < 2:
            return 0.0
        sd = float(np.std(d, ddof=1))
        n = d.size
        se_bb_per_hand = sd / (self.hands_per_unit * math.sqrt(n))
        return 100_000.0 * se_bb_per_hand

    def summary(self) -> str:
        mode = "dup" if self.duplicated else "seq"
        return (f"[match] {self.a_name} vs {self.b_name}  "
                f"mode={mode}  hands={self.n_hands}  "
                f"A={self.a_mbb_per_hand:+.2f} mbb/hand "
                f"({self.a_mbb_per_100:+.0f} mbb/100  "
                f"SE {self.a_std_err_mbb_per_100:.0f})  "
                f"B={-self.a_mbb_per_hand:+.2f} mbb/hand  "
                f"wall={self.elapsed_s:.1f}s")

    @property
    def a_total_actions(self) -> int:
        return int(self.a_slot_counts.sum())

    def slot_freq_overall(self) -> np.ndarray:
        """(NUM_ACTIONS,) array of A's action frequencies across all streets."""
        totals = self.a_slot_counts.sum(axis=0).astype(np.float64)
        n = totals.sum()
        return totals / n if n > 0 else totals

    def slot_summary(self, per_street: bool = True) -> str:
        """Compact multi-line text breakdown of A's action-slot usage.

        Line 1: overall percentages across all decisions.
        Lines 2–5 (if per_street): one per street, only printed if any action
        was taken on that street. Percentages within-street.
        """
        def fmt_row(counts: np.ndarray, label: str) -> str:
            n = int(counts.sum())
            if n == 0:
                return f"[slots-{label:<4}] n=0"
            pct = counts.astype(np.float64) / n
            cells = "  ".join(f"{lab}={p:.02f}"
                              for lab, p in zip(SLOT_LABELS, pct))
            return f"[slots-{label:<4}] n={n:<5} {cells}"

        lines = [fmt_row(self.a_slot_counts.sum(axis=0), "all")]
        if per_street:
            for s, lab in enumerate(STREET_LABELS):
                lines.append(fmt_row(self.a_slot_counts[s], lab))
        return "\n".join(lines)


def _play_one_hand(env, sb_pol, bb_pol, policy_a,
                   rng: np.random.Generator,
                   slot_counts: np.ndarray) -> float:
    """Play env to terminal. Records slot/street when A is to act.
    Returns A's bb-denominated payoff for this hand.

    Precondition: env has just been reset; caller has chosen sb_pol/bb_pol.
    """
    while not env.is_terminal():
        actor = env.to_act()
        obs   = env.observation()
        pol   = sb_pol if actor == pte.Player.SB else bb_pol
        idx   = pol.choose(obs, rng)
        if pol is policy_a:
            # Street one-hot lives at x[104:108] per STATE_ENCODING.md.
            street = int(np.argmax(obs.x[104:108]))
            slot   = int(obs.legal_idx[idx])
            slot_counts[street, slot] += 1
        env.step(idx)

    payoffs = env.payoffs_bb()   # [SB, BB]
    return float(payoffs[0]) if sb_pol is policy_a else float(payoffs[1])


def play_match(policy_a, policy_b,
               n_hands: int,
               base_seed: int = 0xC0FFEE,
               rng: np.random.Generator | None = None,
               duplicated: bool = False) -> MatchResult:
    """Run `n_hands` of head-to-head play between two policies.

    Regular mode (`duplicated=False`): alternate which policy sits at SB every
    hand. Each hand draws its own deck from `base_seed + h`.

    Duplicated mode (`duplicated=True`): play `n_hands / 2` pairs. Each pair
    uses one deck seed but plays the hand twice — once with A at SB, once
    with A at BB. Card-luck variance is largely canceled across the pair.
    `n_hands` is floored to an even number.
    """
    rng = rng or np.random.default_rng(base_seed ^ 0xDEADBEEF)
    env = pte.Env(base_seed)

    slot_counts = np.zeros((NUM_STREETS, NUM_ACTIONS), dtype=np.int64)
    unit_deltas: list[float] = []
    t0 = time.time()

    if duplicated:
        n_pairs = n_hands // 2
        actual_hands = 2 * n_pairs
        for p in range(n_pairs):
            seed = (base_seed + p) & 0xFFFFFFFFFFFFFFFF
            pair_delta = 0.0
            for orientation in (0, 1):
                a_is_sb = (orientation == 0)
                sb_pol = policy_a if a_is_sb else policy_b
                bb_pol = policy_b if a_is_sb else policy_a
                env.reset(seed)
                pair_delta += _play_one_hand(env, sb_pol, bb_pol, policy_a,
                                             rng, slot_counts)
            unit_deltas.append(pair_delta)
        hands_per_unit = 2
    else:
        actual_hands = n_hands
        for h in range(n_hands):
            seed = (base_seed + h) & 0xFFFFFFFFFFFFFFFF
            a_is_sb = (h % 2 == 0)
            sb_pol = policy_a if a_is_sb else policy_b
            bb_pol = policy_b if a_is_sb else policy_a
            env.reset(seed)
            delta = _play_one_hand(env, sb_pol, bb_pol, policy_a,
                                   rng, slot_counts)
            unit_deltas.append(delta)
        hands_per_unit = 1

    a_chips_bb = float(sum(unit_deltas))
    elapsed = time.time() - t0
    return MatchResult(
        n_hands=actual_hands,
        a_bb_total=a_chips_bb,
        b_bb_total=-a_chips_bb,      # zero-sum
        a_name=getattr(policy_a, "name", "A"),
        b_name=getattr(policy_b, "name", "B"),
        elapsed_s=elapsed,
        a_slot_counts=slot_counts,
        a_unit_deltas_bb=np.asarray(unit_deltas, dtype=np.float64),
        hands_per_unit=hands_per_unit,
        duplicated=duplicated,
    )
