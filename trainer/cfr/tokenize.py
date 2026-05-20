"""Token-based state encoding for the transformer AdvNet/PolicyNet.

═══════════════════════════════════════════════════════════════════════════════
WHY THIS EXISTS
═══════════════════════════════════════════════════════════════════════════════

The previous flat-vector encoder (`obs.x`, 816 dims) put hole+board cards
into 104 dims and action history into 680 dims. The Linear(816, 512) stem
in the AdvNet learned to focus on whichever inputs drove the most regret-
target variance — the action history. Cards got drowned out.

Diagnostic confirmed this: a trained AdvNet at iter 80 produced sigma
that varied by only 0.017 across 38 different hole cards. The 52-bit
hole-card region of x existed but the network learned to ignore it.

This module produces a TOKEN SEQUENCE instead. Each card is a full token
slot; each action is HERO/VILLAIN + ACTION + POT_n + STACK_n (4 tokens).
The downstream transformer treats every token equally, so cards have a
fighting chance of influencing the policy via attention.

═══════════════════════════════════════════════════════════════════════════════
VOCABULARY (106 tokens, see project_token_encoding.md)
═══════════════════════════════════════════════════════════════════════════════

  Range     Count  Tokens
  ─────     ─────  ──────
  0–6       7      PAD BOS DECISION SEP_PRE SEP_FLOP SEP_TURN SEP_RIVER
  7–8       2      POS_SB POS_BB        (hero's seat, appears once at start)
  9–10      2      HERO VILLAIN          (relative-to-hero actor prefix)
  11–62     52     card tokens (id = card_value + 11)
  63–73     11     FOLD CHECK_CALL R25 R33 R50 R75 R100 R150 R200 R300 ALL_IN
  74–89     16     POT_0 … POT_15        (log-spaced pot buckets)
  90–105    16     STACK_0 … STACK_15    (eff-stack buckets, fine at low end)

═══════════════════════════════════════════════════════════════════════════════
SEQUENCE TEMPLATE (chronological)
═══════════════════════════════════════════════════════════════════════════════

  [BOS]
  [POS_SB | POS_BB]
  [CARD_hole1] [CARD_hole2]
  [SEP_PRE]
  { [HERO|VILLAIN] [ACTION] [POT_n] [STACK_n] } * n preflop actions
  [SEP_FLOP]                          (only if flop dealt)
  [CARD_f1] [CARD_f2] [CARD_f3]
  { ... }                             (flop actions, 4 tokens each)
  [SEP_TURN]                          (only if turn dealt)
  [CARD_turn]
  { ... }
  [SEP_RIVER]
  [CARD_river]
  { ... }
  [DECISION]

The decision-position embedding (the transformer's hidden state at
[DECISION]) is read by the regret/policy head. Padding is right-padded
to the max-in-batch sequence length with [PAD], masked out of attention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pokertrainer_engine as pte


# ═══════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════

PAD = 0
BOS = 1
DECISION = 2
SEP_PRE = 3
SEP_FLOP = 4
SEP_TURN = 5
SEP_RIVER = 6

POS_SB = 7
POS_BB = 8

HERO = 9
VILLAIN = 10

CARD_OFFSET   = 11   # cards 0..51 → tokens 11..62
ACTION_OFFSET = 63   # ActionType 0..10 → tokens 63..73
POT_OFFSET    = 74   # POT_0..POT_15 → tokens 74..89
STACK_OFFSET  = 90   # STACK_0..STACK_15 → tokens 90..105

VOCAB_SIZE = 106

MAX_SEQ_LEN = 160     # max plausible sequence length
                      # 7 cards + 7 specials + 4 streets * 4 tokens * ~8 actions

# Per-street separator lookup — index by Street enum int value.
_STREET_SEP = (SEP_PRE, SEP_FLOP, SEP_TURN, SEP_RIVER)


# ═══════════════════════════════════════════════════════════════════════════
# BUCKET BOUNDARIES (BB-denominated; both lists are right-exclusive)
# ═══════════════════════════════════════════════════════════════════════════
#
# Pot is log-ish so each bucket transition costs ~similar pot-odds info.
# Effective stack is FINE at low (push-fold land) and COARSE at deep
# (strategy changes slowly when both players have 90+ bb).

# 16 lower-bounds. pot >= POT_BUCKETS_BB[i] but < POT_BUCKETS_BB[i+1] → i.
POT_BUCKETS_BB = (0.0, 2.0, 3.0, 4.5, 6.5, 9.0, 13.0, 18.0,
                  25.0, 35.0, 50.0, 70.0, 100.0, 140.0, 180.0, 220.0)
assert len(POT_BUCKETS_BB) == 16

# 16 lower-bounds for effective stack. STACK_0 specifically = exactly 0.
STACK_BUCKETS_BB = (0.0, 1e-6, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0,
                    17.0, 22.0, 30.0, 40.0, 55.0, 70.0, 85.0, 95.0)
assert len(STACK_BUCKETS_BB) == 16


_BB_CHIPS = 100   # must match pte.BIG_BLIND_CHIPS


def _bucket_index(value: float, bounds: tuple) -> int:
    """Return the largest i such that value >= bounds[i]. Linear scan.
    16 entries → 16 comparisons max — cheap relative to a single embedding
    lookup. Could be `bisect_right(bounds, value) - 1` but the literal
    loop is clearer and not in any hot inner loop."""
    for i in range(len(bounds) - 1, -1, -1):
        if value >= bounds[i]:
            return i
    return 0


def pot_token(pot_chips: int) -> int:
    return POT_OFFSET + _bucket_index(pot_chips / _BB_CHIPS, POT_BUCKETS_BB)


def stack_token(eff_stack_chips: int) -> int:
    """STACK_0 if exactly 0; otherwise bucket lookup."""
    if eff_stack_chips <= 0:
        return STACK_OFFSET
    return STACK_OFFSET + _bucket_index(eff_stack_chips / _BB_CHIPS, STACK_BUCKETS_BB)


def card_token(card: int) -> int:
    return CARD_OFFSET + int(card)


def action_token(action_type) -> int:
    return ACTION_OFFSET + int(action_type)


# ═══════════════════════════════════════════════════════════════════════════
# CORE: state → token sequence
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TokenizedState:
    """Output of `tokenize_state`.

    Used as the input row for AdvNet / PolicyNet forward.
    """
    tokens: np.ndarray          # int64 (L,) — padded later by the buffer/batcher
    decision_pos: int           # index of the [DECISION] token within `tokens`


def tokenize_state(state, hero_seat: int) -> TokenizedState:
    """Produce the token sequence for hero=`hero_seat`'s POV at the current
    decision point.

    Preconditions:
      * `state.to_act == hero_seat`  (we're tokenizing the perspective of
        whoever's about to act).
      * state must NOT be terminal.

    Postconditions:
      * `tokens[-1] == DECISION`.
      * Sequence length is bounded by MAX_SEQ_LEN.

    Note on effective-stack reconstruction: AppliedAction exposes the
    actor's stack_after but not the opponent's at that moment. We track
    both players' stacks in a running map across the history walk —
    starting from state.starting_stacks and applying each action's
    actor-stack delta. The opponent's stack at action k is whatever it
    was after their most recent action (or the starting stack if they
    haven't acted yet).
    """
    if state.is_terminal():
        raise ValueError("tokenize_state called on terminal state")

    tokens: list[int] = []

    # ── Prefix: BOS + hero seat + hole cards ─────────────────────────────
    tokens.append(BOS)
    tokens.append(POS_SB if hero_seat == 0 else POS_BB)
    hole = state.hole[hero_seat]
    tokens.append(card_token(int(hole[0])))
    tokens.append(card_token(int(hole[1])))

    # ── Per-action history ───────────────────────────────────────────────
    # Track per-player stack so we can derive eff-stack at each action.
    # Start from the engine's starting_stacks.
    stack_per_player = [int(s) for s in state.starting_stacks]

    # Emit street separator(s) before any actions of that street happen.
    # We always emit SEP_PRE first because preflop is the first street.
    last_emitted_street = -1
    def _emit_separator_up_to(target_street_idx: int):
        nonlocal last_emitted_street
        while last_emitted_street < target_street_idx:
            last_emitted_street += 1
            tokens.append(_STREET_SEP[last_emitted_street])
            # After preflop, also emit the cards revealed at street start.
            if last_emitted_street == 1:    # FLOP
                for c in state.board[:3]:
                    tokens.append(card_token(int(c)))
            elif last_emitted_street == 2:  # TURN
                tokens.append(card_token(int(state.board[3])))
            elif last_emitted_street == 3:  # RIVER
                tokens.append(card_token(int(state.board[4])))

    # SEP_PRE always emitted before walking history (even if history is empty)
    _emit_separator_up_to(0)

    for act in state.history:
        street_idx = int(act.street)
        _emit_separator_up_to(street_idx)

        # Actor token — relative to hero.
        actor_idx = int(act.actor)
        tokens.append(HERO if actor_idx == hero_seat else VILLAIN)

        # Action type.
        tokens.append(action_token(act.type))

        # Update stack tracking BEFORE building the pot/stack tokens, since
        # AppliedAction.stack_after_chips reflects the actor's post-action
        # stack. The opponent's stack is unchanged by this action.
        stack_per_player[actor_idx] = int(act.stack_after_chips)

        # POT_n and STACK_n at the moment AFTER this action.
        tokens.append(pot_token(int(act.pot_after_chips)))
        eff_stack = min(stack_per_player[0], stack_per_player[1])
        tokens.append(stack_token(eff_stack))

    # ── Catch up street separators to the current state's street ─────────
    # e.g., if we're at flop with no flop action yet, we need to emit
    # SEP_FLOP + the 3 board cards so the model knows the street advanced.
    _emit_separator_up_to(int(state.street))

    # ── DECISION sentinel ────────────────────────────────────────────────
    decision_pos = len(tokens)
    tokens.append(DECISION)

    if len(tokens) > MAX_SEQ_LEN:
        # Should never happen given engine action budgets; truncate
        # defensively (drop earliest history actions). Truncation would
        # only kick in pathological cases (>20 min-raise war per street).
        raise RuntimeError(
            f"tokenize_state produced length {len(tokens)} > MAX_SEQ_LEN={MAX_SEQ_LEN}; "
            f"engine action budget exceeded?")

    return TokenizedState(
        tokens=np.asarray(tokens, dtype=np.int64),
        decision_pos=decision_pos,
    )


# ═══════════════════════════════════════════════════════════════════════════
# BATCHING: pad variable-length sequences for transformer forward
# ═══════════════════════════════════════════════════════════════════════════


def pad_batch(seqs: list[TokenizedState]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad to max length in batch.

    Returns
    -------
    tokens       : int64 (B, L_max)   — padded with PAD=0
    pad_mask     : bool  (B, L_max)   — True at PAD positions
    decision_pos : int64 (B,)         — index of DECISION in each row
    """
    B = len(seqs)
    L_max = max(s.tokens.shape[0] for s in seqs) if B > 0 else 0
    tokens = np.full((B, L_max), PAD, dtype=np.int64)
    pad_mask = np.ones((B, L_max), dtype=bool)
    dpos = np.zeros((B,), dtype=np.int64)
    for i, s in enumerate(seqs):
        L = s.tokens.shape[0]
        tokens[i, :L] = s.tokens
        pad_mask[i, :L] = False
        dpos[i] = s.decision_pos
    return tokens, pad_mask, dpos


# ═══════════════════════════════════════════════════════════════════════════
# HUMAN-READABLE DEBUG (used by tests/diagnostics)
# ═══════════════════════════════════════════════════════════════════════════


def token_to_str(tok: int) -> str:
    """Render a token id as a short label — for debug pretty-printing."""
    if tok == PAD: return "PAD"
    if tok == BOS: return "BOS"
    if tok == DECISION: return "DECISION"
    if tok == SEP_PRE: return "SEP_PRE"
    if tok == SEP_FLOP: return "SEP_FLOP"
    if tok == SEP_TURN: return "SEP_TURN"
    if tok == SEP_RIVER: return "SEP_RIVER"
    if tok == POS_SB: return "POS_SB"
    if tok == POS_BB: return "POS_BB"
    if tok == HERO: return "HERO"
    if tok == VILLAIN: return "VILLAIN"
    if CARD_OFFSET <= tok < CARD_OFFSET + 52:
        card = tok - CARD_OFFSET
        rank = card >> 2
        suit = card & 3
        return f"C_{'23456789TJQKA'[rank]}{'cdhs'[suit]}"
    if ACTION_OFFSET <= tok < ACTION_OFFSET + 11:
        labels = ("F", "C", "R25", "R33", "R50", "R75",
                  "R100", "R150", "R200", "R300", "AI")
        return labels[tok - ACTION_OFFSET]
    if POT_OFFSET <= tok < POT_OFFSET + 16:
        return f"POT_{tok - POT_OFFSET}"
    if STACK_OFFSET <= tok < STACK_OFFSET + 16:
        return f"STACK_{tok - STACK_OFFSET}"
    return f"???_{tok}"


def render_sequence(t: TokenizedState) -> str:
    """One-line human-readable rendering of the full token sequence."""
    return " ".join(token_to_str(int(tok)) for tok in t.tokens)
