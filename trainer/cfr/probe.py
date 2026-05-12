"""Training-time learning probes for cfr_coro.

═══════════════════════════════════════════════════════════════════════════════
WHAT THIS ANSWERS
═══════════════════════════════════════════════════════════════════════════════

Self-play loss curves drop monotonically even when the policy is collapsing
into nonsense (the 1M-step trap that motivated the project's direction shift).
What we actually want during training is a cheap "is this thing learning
*about poker*?" signal.

The cheapest such signal: hand the net a known-good and known-bad hole, let
it play against a uniform-random opponent, and look at chip delta + action
mix. If the AdvNet has any hand-strength awareness, AA should crush random
and 72o should crush a lot less.

  random-vs-random with AA wins roughly +3 to +4 bb/hand (pure equity).
  A coherent aggressive AA strategy hits +5 to +7 bb/hand against random.
  72o random-vs-random is roughly flat; the right strategy is "fold to
  any pressure" and the bb count depends on how often the opponent bets.

The DELTA between the AA bb count and the 72o bb count is the diagnostic —
a flat policy that ignores hole cards gives ~0 delta; a hand-aware policy
gives several bb/hand of delta.

═══════════════════════════════════════════════════════════════════════════════
WHY REJECTION SAMPLING (NOT FORCED HOLE)
═══════════════════════════════════════════════════════════════════════════════

`HUState.hole` is exposed read-write in pybind, but the engine's deck state
is loaded at `deal()` time. Overwriting hole afterwards risks dealing the
same card on the board on later streets — silent corruption rather than a
hard error. Rejection-sampling is correct without an engine change, and:
  * AA shows up ~1/221 deals (0.45%)
  * 72o (offsuit) shows up at 12/1326 deals (0.9%)
finding 400 of each is well under 100k `env.reset()` calls (<100ms).

═══════════════════════════════════════════════════════════════════════════════
COST
═══════════════════════════════════════════════════════════════════════════════

Per scenario (400 hands, ~5 decisions/hand, B=1 GPU forward at ~250µs):
  ≈ 400 × 5 × 250µs = 500ms of GPU inference
  + 50ms of seed-trial rejection
  + 50ms of engine play
≈ 1 second per scenario, 2 seconds for both AA and 72o. Negligible against
the ~3-minute cluster iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch

import pokertrainer_engine as pte

from evaluate.policies import CFRAdvPolicy, RandomPolicy
from evaluate.match import NUM_ACTIONS, SLOT_LABELS


# ═══════════════════════════════════════════════════════════════════════════
# HOLE-CARD PREDICATES
# ═══════════════════════════════════════════════════════════════════════════
#
# Card encoding from engine/src/card.h:
#   card = rank << 2 | suit
#   rank 0..12 : 2..A   (12 = Ace)
#   suit 0..3  : clubs, diamonds, hearts, spades
#
# state.hole[player] is a 2-element sequence of these uint8 card indices.
# ═══════════════════════════════════════════════════════════════════════════


def _ranks_of_hole(hole_pair) -> tuple[int, int]:
    """Return (high_rank, low_rank), descending — convenient for hole patterns."""
    r0 = int(hole_pair[0]) >> 2
    r1 = int(hole_pair[1]) >> 2
    return (max(r0, r1), min(r0, r1))


def _suits_of_hole(hole_pair) -> tuple[int, int]:
    return (int(hole_pair[0]) & 3, int(hole_pair[1]) & 3)


def is_pocket_aces(hole_pair) -> bool:
    """AA: both cards rank 12. Suits are guaranteed to differ (one deck)."""
    return _ranks_of_hole(hole_pair) == (12, 12)


def is_seven_two_offsuit(hole_pair) -> bool:
    """72o: ranks {7, 2} ≡ rank-codes {5, 0}, different suits.

    72s (same suit) is excluded — it has materially better equity than
    72o (~33% vs random hand on the flop, vs 28% for 72o). Keeping the
    probe narrowly on 72o makes the "worst hand in HU" interpretation
    actually mean what people mean by it.
    """
    if _ranks_of_hole(hole_pair) != (5, 0):
        return False
    s0, s1 = _suits_of_hole(hole_pair)
    return s0 != s1


# ═══════════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ProbeResult:
    label: str
    n_hands: int
    seeds_tried: int
    net_bb_total: float        # net's chip delta in BB, summed over all hands and both seats
    elapsed_s: float
    preflop_slot_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_ACTIONS, dtype=np.int64))

    @property
    def mbb_per_hand(self) -> float:
        return 1000.0 * self.net_bb_total / max(1, self.n_hands)

    def preflop_freq_str(self) -> str:
        """Compact preflop action distribution, e.g. 'F=0.05  C=0.30  R3x=...'."""
        n = int(self.preflop_slot_counts.sum())
        if n == 0:
            return "n=0"
        pct = self.preflop_slot_counts.astype(np.float64) / n
        return "  ".join(f"{lab}={p:.02f}" for lab, p in zip(SLOT_LABELS, pct))


# ═══════════════════════════════════════════════════════════════════════════
# PLAY-ONE-HAND HELPER
# ═══════════════════════════════════════════════════════════════════════════


def _play_one(env,
              net_policy: CFRAdvPolicy,
              opp_policy: RandomPolicy,
              net_seat: int,
              rng: np.random.Generator,
              preflop_slot_counts: np.ndarray) -> float:
    """Play `env` to terminal. Net sits at `net_seat`; opponent at the other.

    Records the net's preflop action slot. Returns net's payoff in BB.

    NOTE: street 0 (preflop) is identified by argmax of x[104:108] — same
    convention the match runner uses (evaluate/match.py:148). Postflop
    actions are not recorded; they're noisy and not the diagnostic we want.
    """
    while not env.is_terminal():
        actor = int(env.to_act())
        obs = env.observation()
        if actor == net_seat:
            idx = net_policy.choose_with_seat(obs, net_seat, rng)
            street = int(np.argmax(obs.x[104:108]))
            if street == 0:
                slot = int(obs.legal_idx[idx])
                preflop_slot_counts[slot] += 1
        else:
            idx = opp_policy.choose(obs, rng)
        env.step(idx)

    return float(env.payoffs_bb()[net_seat])


# ═══════════════════════════════════════════════════════════════════════════
# PROBE DRIVER
# ═══════════════════════════════════════════════════════════════════════════


def run_probe(adv_nets: list[torch.nn.Module],
              device: torch.device,
              label: str,
              predicate,
              *,
              n_hands: int = 400,
              max_searches: int = 200_000,
              base_seed: int = 0xB10ED) -> ProbeResult:
    """Rejection-sample seeds until `n_hands` matches have been played.

    Seat-alternates: hand 0 puts the net at SB (seat 0), hand 1 at BB (seat 1),
    and so on. The opposing seat plays uniform random. This makes the chip
    delta a mix of "your hole + your position" and is robust to position
    asymmetry — useful because AA from BB plays very differently than AA
    from SB facing an open.

    `max_searches` is a safety: AA/72o show up frequently enough that 200k
    is ~5× the expected budget for 400 hands. If we hit it without filling
    n_hands, we return what we have and the caller can see "n_hands < target"
    in the result.
    """
    t0 = time.time()
    net_policy = CFRAdvPolicy(adv_nets, device, stochastic=True, name="cfr_probe")
    opp_policy = RandomPolicy()
    env = pte.Env(base_seed)
    rng_actions = np.random.default_rng(base_seed ^ 0xCAFEFACE)

    pre_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    net_bb_total = 0.0
    n_played = 0
    n_seeds_tried = 0

    while n_played < n_hands and n_seeds_tried < max_searches:
        seed_h = (base_seed + n_seeds_tried) & 0xFFFFFFFFFFFFFFFF
        env.reset(seed_h)
        n_seeds_tried += 1
        # Whose hole we filter on depends on which seat the net will occupy
        # this hand. Alternating SB/BB keeps the report symmetric.
        net_seat = n_played % 2
        if not predicate(env.state().hole[net_seat]):
            continue
        net_bb_total += _play_one(env, net_policy, opp_policy,
                                  net_seat, rng_actions, pre_counts)
        n_played += 1

    return ProbeResult(
        label=label,
        n_hands=n_played,
        seeds_tried=n_seeds_tried,
        net_bb_total=net_bb_total,
        elapsed_s=time.time() - t0,
        preflop_slot_counts=pre_counts,
    )


def run_default_probes(adv_nets: list[torch.nn.Module],
                       device: torch.device,
                       n_hands: int = 400,
                       base_seed: int = 0xB10ED) -> list[ProbeResult]:
    """Run the AA and 72o probes back-to-back.

    The two probes use DIFFERENT base seeds so the same hand index isn't
    forced to use the same seed search across scenarios — otherwise the
    rejection-sampler would skip the same seeds in both, and the comparison
    might look biased if some seed pattern happens to be unusual.
    """
    return [
        run_probe(adv_nets, device, "AA ", is_pocket_aces,
                  n_hands=n_hands, base_seed=base_seed),
        run_probe(adv_nets, device, "72o", is_seven_two_offsuit,
                  n_hands=n_hands, base_seed=base_seed ^ 0xA5A5),
    ]


def format_probe_line(r: ProbeResult, iter_t: int | None = None) -> str:
    """Single-line summary suitable for the training log."""
    prefix = f"[probe iter={iter_t}]" if iter_t is not None else "[probe]"
    return (f"  {prefix} {r.label}  n={r.n_hands:<4} "
            f"({r.seeds_tried:>6} seeds, {r.elapsed_s:>4.1f}s)  "
            f"net={r.mbb_per_hand:+7.0f} mbb/hand  "
            f"preflop: {r.preflop_freq_str()}")
