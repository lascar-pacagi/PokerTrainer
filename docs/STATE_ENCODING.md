# State Encoding Specification

**Status:** v0.5 (reverse-chronological history with per-street truncation
flag, heads-up NLHE). Bit-identical outputs required from
`engine/src/encoder.cpp` and any Python reproduction. Any change here
requires updating both and re-running `engine/tests/test_encoder.cpp` and
the trainer smokes.

## What changed from v0.4

- **Within each street's sub-block, slot 0 is now the MOST RECENT action of
  that street** (not the kth chronological action). Slot 1 is the
  next-most-recent, etc. Padded slots remain zero.
- **Per-street truncation is silent**, not fatal. When a street has more
  actions than its slot budget, the OLDEST actions are dropped — the most
  recent `budget` actions still occupy slots 0..budget-1 — and a new
  per-street truncation bit fires in the static block.
- **New static-block field `HIST_TRUNCATED[4]`** at offset 132. Bit `i`
  is 1.0 iff street `i` lost actions to truncation.
- `STATIC_DIM` 132 → 136. `X_DIM` 812 → 816. `HIST_FEAT` unchanged (20).
- **Rationale for slot-0 = most recent**: the slot's interpretation is
  stable regardless of how deep the action sequence got, so the MLP's
  per-slot weights specialize to a single semantics ("what just
  happened"). Under v0.4's chronological ordering, slot N-1's meaning
  varied with the action count.
- **Rationale for silent truncation + flag**: high-ε exploration during
  early training can produce min-raise wars longer than the budget. v0.4
  threw at that point, killing the training run. v0.5 keeps the most
  informationally relevant rows (recent actions) and tells the network via
  the static-block flag that older context is unobservable.

## Card index convention

A card is a `uint8_t` in `[0, 52)`. Layout: `card = rank * 4 + suit`, where:

- `rank ∈ [0, 12]`: 0=deuce, 1=trey, ..., 8=ten, 9=jack, 10=queen, 11=king, 12=ace
- `suit ∈ [0, 3]`: 0=clubs, 1=diamonds, 2=hearts, 3=spades

This matches the hand evaluator's `make_card(rank, suit)` indexing.

52-dim binary encoding: `v[card] = 1` iff the card is present, else 0.

## Action discretization

11 discrete action slots, indexed 0..10:

| idx | name           | bet_to                                     |
|-----|----------------|--------------------------------------------|
| 0   | FOLD           | —                                          |
| 1   | CHECK_CALL     | match the current to-call amount           |
| 2   | RAISE_25       | bet_to = current_bet + 0.25 * (pot + to_call) |
| 3   | RAISE_33       | bet_to = current_bet + 0.33 * (pot + to_call) |
| 4   | RAISE_50       | bet_to = current_bet + 0.50 * (pot + to_call) |
| 5   | RAISE_75       | bet_to = current_bet + 0.75 * (pot + to_call) |
| 6   | RAISE_100      | pot-sized raise after call                 |
| 7   | RAISE_150      | 1.5× pot                                   |
| 8   | RAISE_200      | 2.0× pot                                   |
| 9   | RAISE_300      | 3.0× pot                                   |
| 10  | ALL_IN         | shove effective stack                      |

`current_bet` is the highest amount invested **this street** (the level a call
matches); it equals `to_call` only *before* you have put chips in this street,
so the two are **not** interchangeable after a prior bet/raise. The pot fraction
is applied to the pot *after* the hypothetical call. The implementation is
`HUState::raise_to_from_fraction` (`engine/src/game_hu.cpp`).

Raises below legal min-raise are snapped up; above effective stack are snapped
down to all-in. Illegal actions are masked at the Env level. Note: on small
pots, RAISE_25 often snaps up to min-raise and becomes indistinguishable from
RAISE_33 — the engine de-duplicates and only the smaller of the colliding
slots appears in the legal set. The legal-actions mask in `x` (see below)
makes this visible to the network.

## Per-action encoding `a` (11 floats per row)

| slice       | dims | meaning                                      |
|-------------|------|----------------------------------------------|
| action_type | 11   | pure one-hot over the 11 ActionType slots    |

That's the entire row. Exactly one of the 11 floats is 1.0; the rest are 0.0.
No state-dependent scalars. Scoring a decision builds an `(n_legal, 11)`
tensor and the network processes each row alongside the same `x`.

## Current-state vector `x` (816 floats)

Four regions: **static state** (121), **legal-actions mask** (11),
**per-street truncation flags** (4), **reverse-chronological action
history** (680 = 34 × 20).

### Static state — offsets 0..120 (121 floats)

| offset | slice                    | dims | meaning                                               |
|--------|--------------------------|------|-------------------------------------------------------|
|   0    | hole_cards               | 52   | binary, 2 bits set                                    |
|  52    | board_cards              | 52   | binary, 0/3/4/5 bits set by street                    |
| 104    | street                   |  4   | one-hot {preflop, flop, turn, river}                  |
| 108    | position                 |  2   | one-hot {OOP, IP} — SB is IP postflop, OOP preflop    |
| 110    | pot_bb                   |  1   | current pot in bb                                     |
| 111    | our_stack_bb             |  1   | bb                                                    |
| 112    | villain_stack_bb         |  1   | bb                                                    |
| 113    | effective_stack_bb       |  1   | min(p0_total, p1_total) in bb                         |
| 114    | spr                      |  1   | effective_stack / pot (effective_stack if pot==0)     |
| 115    | to_call_bb               |  1   | amount to call in bb                                  |
| 116    | to_call_frac_pot         |  1   | to_call / pot (0 if pot empty)                        |
| 117    | our_invested_this_street |  1   | bb put in this street by us                           |
| 118    | vil_invested_this_street |  1   | bb put in this street by villain                      |
| 119    | n_actions_this_street    |  1   | count, clipped to 6                                   |
| 120    | street_first_to_act      |  1   | 1 if we're first to act this street                   |

### Legal-actions mask — offsets 121..131 (11 floats)

`x[121 + k] = 1.0` iff `ActionType(k)` is legal in the current state, else
`0.0`. Mirrors `s.legal_actions_mask()`. Bit `k` corresponds to the same
action index used everywhere else in the encoding.

### Per-street truncation flags — offsets 132..135 (4 floats)

`x[132 + s] = 1.0` iff street `s`'s action count exceeded its slot budget
(see below) and the encoder dropped its oldest actions.
`s` is 0=preflop, 1=flop, 2=turn, 3=river. When 0, the visible history
rows for that street are exhaustive; when 1, they are the most recent
budget actions only.

### Reverse-chronological action history — offsets 136..815 (34 rows × 20 feat = 680 floats)

The history block is partitioned into four contiguous per-street sub-blocks:

| street  | slot count | start offset | end offset (exclusive) |
|---------|-----------:|-------------:|-----------------------:|
| preflop |         10 |          136 |                    336 |
| flop    |          8 |          336 |                    496 |
| turn    |          8 |          496 |                    656 |
| river   |          8 |          656 |                    816 |

**Within a sub-block, slot 0 is the MOST RECENT action of that street**;
slot 1 is the next-most-recent; etc. So slot k is "the kth-from-last
action of this street." This means slot 0's interpretation is stable
("what just happened on this street") regardless of how many actions
preceded.

If a street has fewer actions than its slot budget, the trailing slots are
zero-padded; their `is_real` bit (offset 0 within the row) is 0.

If a street has *more* actions than its slot budget, the oldest are
dropped (the most-recent `budget` survive in slots 0..budget-1) and the
matching bit in the truncation block above is set to 1.0.

Slot budgets were sized to cover the empirical worst case under uniform-
random play (100k hands → max `[9, 7, 7, 6]`); the budgets `[10, 8, 8, 8]`
sit a small margin above that. Truncation should be a tail event — it
fires when high-ε exploration produces min-raise wars longer than typical
play.

Each row's 20 floats:

| within-row offset | dims | meaning                                                |
|-------------------|------|--------------------------------------------------------|
| 0                 |  1   | **is_real** — 1.0 for populated row, 0.0 for padding   |
| 1–11              | 11   | action_type one-hot (same indexing as `a`)             |
| 12                |  1   | bet_to_bb                                              |
| 13                |  1   | bet_frac_pot (vs pot BEFORE this action; 0 if pot==0)  |
| 14                |  1   | pot_after_bb                                           |
| 15                |  1   | stack_after_bb (actor's stack post-action, populated)  |
| 16                |  1   | was_all_in (1 iff this action put the actor all-in)    |
| 17                |  1   | is_raise                                               |
| 18                |  1   | actor_was_us (1 if actor == current to_act player)     |
| 19                |  1   | actor_was_villain (1 otherwise)                        |

Note: the actor one-hot is resolved **at observation time** — `was_us` is
"were *you* the actor", i.e. `act.actor == state.to_act`. When the hand ends,
`encode()` refuses to run (terminal states are never observed).

**Total: 121 + 11 + 4 + 680 = 816 floats.** The offsets above are
**load-bearing**.

## Tensor shapes (Python side)

```python
obs.x         : np.ndarray[shape=(816,),        dtype=float32]
obs.a         : np.ndarray[shape=(n_legal, 11), dtype=float32]
obs.legal     : list[ActionType] of length n_legal
obs.legal_idx : np.ndarray[shape=(n_legal,),    dtype=int8]
```

Module-level constants (`pokertrainer_engine`):

```
X_DIM, A_DIM, HIST_MAX, HIST_FEAT, STATIC_DIM, LEGAL_MASK_DIM,
HIST_TRUNC_DIM, NUM_ACTIONS,
X_OFF_LEGAL_MASK, X_OFF_HIST_TRUNCATED,
X_OFF_PREFLOP, X_OFF_FLOP, X_OFF_TURN, X_OFF_RIVER,
PREFLOP_SLOTS, FLOP_SLOTS, TURN_SLOTS, RIVER_SLOTS,
STREET_SLOTS = (10, 8, 8, 8),
STREET_OFFSETS = (X_OFF_PREFLOP, X_OFF_FLOP, X_OFF_TURN, X_OFF_RIVER),
```

C ABI exposes the same via `pt_x_dim()` / `pt_hist_max()` / `pt_hist_feat()` /
`pt_x_off_street(street)` / `pt_street_slots(street)`.
