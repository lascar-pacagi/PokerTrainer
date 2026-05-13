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
# PREFLOP SITUATION BUCKETING
# ═══════════════════════════════════════════════════════════════════════════
#
# Splitting a single "preflop action mix" line by the betting situation the
# net is *facing* makes the diagnostic far sharper. AA "raising 70% of the
# time" is uninformative if we don't know which 70% — is it raising opens,
# folding to 3-bets, what? Per-situation breakdowns answer this.
#
# We bucket every preflop decision into one of four spots using only the
# engine's `invested_this_street` (= chips voluntarily committed by each
# player on this street, including the blinds). No action-history walk
# needed:
#
#   open      — SB's very first preflop decision (only chip in is the SB blind).
#   vs_limp   — BB to act, no chips to call (SB completed without raising).
#   vs_raise  — actor faces a raise but has not voluntarily raised yet itself.
#               Covers BB facing SB's open, and SB facing BB's limp-raise.
#   vs_3bet+  — actor already raised this street and is now facing a re-raise.
#               Covers SB facing a 3-bet, BB facing a 4-bet, and beyond.
#
# Blind sizes from engine/src/game_hu.h: SB=50 chips (0.5 bb), BB=100 chips.
# ═══════════════════════════════════════════════════════════════════════════

PRE_SITUATIONS = ("open", "vs_limp", "vs_raise", "vs_3bet+")
_PRE_SITUATION_IDX = {s: i for i, s in enumerate(PRE_SITUATIONS)}

_SB_BLIND_CHIPS = 50    # must match HUState::SMALL_BLIND_CHIPS
_BB_BLIND_CHIPS = 100   # must match HUState::BIG_BLIND_CHIPS


def _categorize_preflop(state, actor: int) -> str:
    """Bucket a preflop decision; see PRE_SITUATIONS block comment above."""
    actor_inv = int(state.invested_this_street[actor])
    other_inv = int(state.invested_this_street[1 - actor])
    to_call = other_inv - actor_inv

    # actor==0 → SB, actor==1 → BB (matches pte.Player enum).
    if actor == 0 and actor_inv == _SB_BLIND_CHIPS:
        # SB's only chip in is the forced blind ⇒ no voluntary action yet.
        return "open"
    if actor == 1 and to_call == 0:
        # BB faces nothing to call ⇒ SB limp-completed without raising.
        return "vs_limp"
    # From here, to_call > 0 — we're facing a raise of some level.
    if actor_inv == _BB_BLIND_CHIPS:
        # Actor's chips in = 1bb. For BB this is "still just the blind"
        # (never acted), for SB this is "limped earlier" — either way,
        # there's exactly one raise on the board.
        return "vs_raise"
    # Actor invested > 1bb voluntarily ⇒ already raised this street ⇒ this
    # decision is facing a re-raise.
    return "vs_3bet+"


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
    # Per-situation, per-slot preflop action counts.
    # Shape (len(PRE_SITUATIONS), NUM_ACTIONS).
    preflop_slot_counts: np.ndarray = field(
        default_factory=lambda: np.zeros((len(PRE_SITUATIONS), NUM_ACTIONS),
                                         dtype=np.int64))

    @property
    def mbb_per_hand(self) -> float:
        return 1000.0 * self.net_bb_total / max(1, self.n_hands)

    def preflop_breakdown_lines(self, indent: str = "       ") -> list[str]:
        """One line per non-empty preflop situation, percentages across action slots.

        Empty buckets (e.g. vs_3bet+ never reached) are dropped to keep the
        log compact. The bucket order follows PRE_SITUATIONS (open, vs_limp,
        vs_raise, vs_3bet+), which is the natural sequence of preflop
        re-raises and reads top-to-bottom like the betting tree.
        """
        lines: list[str] = []
        col_label_w = max(len(s) for s in PRE_SITUATIONS)
        for i, sit in enumerate(PRE_SITUATIONS):
            counts = self.preflop_slot_counts[i]
            n = int(counts.sum())
            if n == 0:
                continue
            pct = counts.astype(np.float64) / n
            cells = " ".join(f"{lab}={p:.02f}"
                             for lab, p in zip(SLOT_LABELS, pct))
            lines.append(f"{indent}{sit:<{col_label_w}}  n={n:<4} {cells}")
        return lines


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

    For each preflop decision by the net, bucket by betting situation
    (open / vs_limp / vs_raise / vs_3bet+) and tally the action slot.
    Returns net's payoff in BB.

    NOTE: street 0 (preflop) is identified by argmax of x[104:108] — same
    convention the match runner uses (evaluate/match.py:148). Postflop
    actions are not recorded; they're noisy and not the diagnostic we want.

    The situation lookup uses `env.state()` BEFORE the action is applied,
    so the bucket reflects what the actor was facing when it chose.
    """
    while not env.is_terminal():
        actor = int(env.to_act())
        obs = env.observation()
        if actor == net_seat:
            idx = net_policy.choose_with_seat(obs, net_seat, rng)
            street = int(np.argmax(obs.x[104:108]))
            if street == 0:
                slot = int(obs.legal_idx[idx])
                sit = _categorize_preflop(env.state(), net_seat)
                preflop_slot_counts[_PRE_SITUATION_IDX[sit], slot] += 1
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

    pre_counts = np.zeros((len(PRE_SITUATIONS), NUM_ACTIONS), dtype=np.int64)
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
    """Multi-line summary for the training log.

    Line 1: scenario header — label, hand count, seed budget, wall, mbb/hand.
    Lines 2+: one per non-empty preflop situation bucket (open, vs_limp,
              vs_raise, vs_3bet+), showing the per-slot action mix the net
              chose in that situation.
    """
    prefix = f"[probe iter={iter_t}]" if iter_t is not None else "[probe]"
    header = (f"  {prefix} {r.label}  n={r.n_hands:<4} "
              f"({r.seeds_tried:>6} seeds, {r.elapsed_s:>4.1f}s)  "
              f"net={r.mbb_per_hand:+7.0f} mbb/hand")
    return "\n".join([header, *r.preflop_breakdown_lines()])
