"""Unit + integration tests for cfr/tokenize.py.

Scenarios covered:
  1. Fresh deal, SB to act preflop — minimum-length sequence.
  2. BB facing SB's open — verify [VILLAIN][R100][POT_n][STACK_n] is emitted.
  3. Postflop spot (flop action) — verify SEP_FLOP + 3 board cards appear.
  4. Vocab IDs are all within [0, VOCAB_SIZE).
  5. Hole-card sensitivity: AA vs 72o sequences differ only in 2 card slots.
  6. pad_batch round-trip: variable-length seqs → uniform tensor.

Run:
    PYTHONPATH=engine/build:trainer python trainer/tests/test_tokenize.py
"""
from __future__ import annotations

import numpy as np

import pokertrainer_engine as pte

from cfr.tokenize import (
    VOCAB_SIZE, MAX_SEQ_LEN,
    PAD, BOS, DECISION, SEP_PRE, SEP_FLOP, SEP_TURN, SEP_RIVER,
    POS_SB, POS_BB, HERO, VILLAIN,
    CARD_OFFSET, ACTION_OFFSET, POT_OFFSET, STACK_OFFSET,
    tokenize_state, pad_batch, render_sequence, token_to_str,
)


def _ranks(hole_pair) -> tuple[int, int]:
    return tuple(sorted((int(hole_pair[0]) >> 2,
                         int(hole_pair[1]) >> 2), reverse=True))


def _find_seed_with(env, predicate, max_tries: int = 5000) -> int:
    for s in range(max_tries):
        env.reset(s)
        if predicate(env.state().hole[0]):
            return s
    raise RuntimeError("no matching seed")


def test_fresh_deal_sb_to_act():
    """Right after env.reset(), SB is to act; sequence is minimum-length:
    [BOS, POS_SB, hole1, hole2, SEP_PRE, DECISION]."""
    env = pte.Env(42)
    env.reset(42)
    assert int(env.state().to_act) == 0   # SB
    t = tokenize_state(env.state(), hero_seat=0)
    assert t.tokens[0] == BOS
    assert t.tokens[1] == POS_SB
    assert CARD_OFFSET <= t.tokens[2] < CARD_OFFSET + 52
    assert CARD_OFFSET <= t.tokens[3] < CARD_OFFSET + 52
    assert t.tokens[4] == SEP_PRE
    assert t.tokens[5] == DECISION
    assert t.decision_pos == 5
    assert len(t.tokens) == 6
    print(f"  fresh-deal seq: {render_sequence(t)}")


def test_bb_facing_open():
    """SB opens R100, BB to act. Sequence:
    [BOS, POS_BB, hole1, hole2, SEP_PRE, VILLAIN, R100, POT_n, STACK_n, DECISION]."""
    env = pte.Env(42)
    env.reset(42)
    env.step_action(pte.ActionType.RAISE_100)
    assert int(env.state().to_act) == 1   # BB
    t = tokenize_state(env.state(), hero_seat=1)
    # Layout check
    assert t.tokens[0] == BOS
    assert t.tokens[1] == POS_BB
    # hole cards at 2,3
    assert t.tokens[4] == SEP_PRE
    assert t.tokens[5] == VILLAIN              # SB acted
    assert t.tokens[6] == ACTION_OFFSET + int(pte.ActionType.RAISE_100)
    assert POT_OFFSET <= t.tokens[7] < POT_OFFSET + 16
    assert STACK_OFFSET <= t.tokens[8] < STACK_OFFSET + 16
    assert t.tokens[9] == DECISION
    assert t.decision_pos == 9
    print(f"  BB-facing-open seq: {render_sequence(t)}")


def test_postflop_layout():
    """After SB opens / BB calls / BB acts on flop, the sequence should
    contain SEP_FLOP + 3 board cards before any flop action emission."""
    env = pte.Env(42)
    env.reset(42)
    env.step_action(pte.ActionType.RAISE_100)
    env.step_action(pte.ActionType.CHECK_CALL)
    # Now on flop, BB to act first postflop in HU.
    assert env.state().street == pte.Street.FLOP
    assert int(env.state().to_act) == 1   # BB
    t = tokenize_state(env.state(), hero_seat=1)
    # Expect SEP_FLOP then 3 card tokens then DECISION.
    seq = list(int(x) for x in t.tokens)
    sf_idx = seq.index(SEP_FLOP)
    # Next 3 tokens are board cards
    for k in range(1, 4):
        assert CARD_OFFSET <= seq[sf_idx + k] < CARD_OFFSET + 52, \
            f"flop card not at offset {sf_idx + k}, got {seq[sf_idx + k]}"
    # Then DECISION as last
    assert seq[-1] == DECISION
    print(f"  postflop seq: {render_sequence(t)}")


def test_vocab_bounds():
    """No token id should ever escape [0, VOCAB_SIZE)."""
    env = pte.Env(42)
    # A grab-bag of scenarios — preflop, after one open, postflop.
    scenarios = []
    env.reset(42)
    scenarios.append((env.state(), 0))
    env.step_action(pte.ActionType.RAISE_100)
    scenarios.append((env.state(), 1))
    env.step_action(pte.ActionType.RAISE_300)
    scenarios.append((env.state(), 0))
    env.step_action(pte.ActionType.CHECK_CALL)
    scenarios.append((env.state(), 1))

    for state, seat in scenarios:
        if int(state.to_act) != seat:
            continue
        t = tokenize_state(state, seat)
        assert (t.tokens >= 0).all()
        assert (t.tokens < VOCAB_SIZE).all(), \
            f"token id ≥ {VOCAB_SIZE} found in {t.tokens}"
        assert t.tokens[t.decision_pos] == DECISION
    print(f"  vocab bounds OK across {len(scenarios)} scenarios")


def test_hole_card_sensitivity():
    """The whole point of this refactor: two hole cards (AA vs 72o) at
    the same SB-open spot produce sequences that differ ONLY in the
    hole-card token positions. No other token should change."""
    env = pte.Env(0xC0FFEE)
    s_aa = _find_seed_with(env, lambda h: _ranks(h) == (12, 12))
    s_72o = _find_seed_with(env, lambda h: _ranks(h) == (5, 0)
                            and (int(h[0]) & 3) != (int(h[1]) & 3))
    env.reset(s_aa)
    t_aa = tokenize_state(env.state(), 0)
    env.reset(s_72o)
    t_72o = tokenize_state(env.state(), 0)
    assert t_aa.tokens.shape == t_72o.tokens.shape
    diff_idx = np.where(t_aa.tokens != t_72o.tokens)[0]
    # Hole tokens at positions 2 and 3.
    assert set(diff_idx) == {2, 3}, f"unexpected diff positions: {diff_idx}"
    print(f"  AA seq:  {render_sequence(t_aa)}")
    print(f"  72o seq: {render_sequence(t_72o)}")
    print(f"  diff only at positions {list(diff_idx)} (hole-card slots)")


def test_pad_batch_round_trip():
    """Batch a mix of short and long sequences, verify padding + decision_pos."""
    env = pte.Env(42)
    seqs = []
    env.reset(1)
    seqs.append(tokenize_state(env.state(), 0))          # fresh, length 6
    env.reset(2)
    env.step_action(pte.ActionType.RAISE_100)
    seqs.append(tokenize_state(env.state(), 1))          # BB facing open
    env.reset(3)
    env.step_action(pte.ActionType.RAISE_100)
    env.step_action(pte.ActionType.CHECK_CALL)
    seqs.append(tokenize_state(env.state(), 1))          # postflop BB

    tokens, pad_mask, dpos = pad_batch(seqs)
    assert tokens.shape[0] == 3
    L_max = tokens.shape[1]
    assert L_max == max(s.tokens.shape[0] for s in seqs)
    # Each row's decision_pos should land on a DECISION token.
    for i, s in enumerate(seqs):
        assert tokens[i, dpos[i]] == DECISION
        # PAD positions are masked out
        assert pad_mask[i, s.tokens.shape[0]:].all()
        # Non-PAD positions are not masked
        assert (~pad_mask[i, :s.tokens.shape[0]]).all()
    print(f"  batch shape: tokens={tokens.shape}, dpos={list(dpos)}")


def main() -> None:
    print("[test_tokenize] running...")
    test_fresh_deal_sb_to_act()
    test_bb_facing_open()
    test_postflop_layout()
    test_vocab_bounds()
    test_hole_card_sensitivity()
    test_pad_batch_round_trip()
    print("[test_tokenize] all tests passed ✓")


if __name__ == "__main__":
    main()
