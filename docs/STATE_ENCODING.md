# State Encoding Specification

**Status:** v0.4 (fixed-position history, heads-up NLHE). Bit-identical
outputs required from `engine/src/encoder.cpp` and any Python reproduction
of the same encoding. Any change here requires updating both and re-running
`engine/tests/test_encoder.cpp` and the trainer smokes.

## What changed from v0.3

- **Action history is now partitioned into four fixed-position sub-blocks**:
  preflop, flop, turn, river. Slot `k` of a sub-block always means "the
  `k`th action of that street." A given row no longer mixes streets across
  decision contexts. This lets the MLP's per-slot weights specialize cleanly
  ("row k of the flop block" is one function, not a context-dependent
  dispatch).
- **Per-row street one-hot dropped** (slot index implies street).
- **Per-row `pos_norm` dropped** (slot index implies position within the
  street).
- `HIST_FEAT` 25 → **20** (-5 floats per row).
- `HIST_MAX` 24 → **34** (10 + 8 + 8 + 8 across streets).
- `STATIC_DIM` 132 (unchanged). `X_DIM` 732 → **812**. `A_DIM` 11 (unchanged).
- **Per-street action budgets** are sized comfortably above the empirical
  worst case under uniform-random play (100k random hands → max
  `[9, 7, 7, 6]` per street). If actual play ever exceeds a budget,
  `encode()` throws — fail-fast, not silent truncation.

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
| 2   | RAISE_25       | bet_to = to_call + 0.25 * (pot + to_call)  |
| 3   | RAISE_33       | bet_to = to_call + 0.33 * (pot + to_call)  |
| 4   | RAISE_50       | bet_to = to_call + 0.50 * (pot + to_call)  |
| 5   | RAISE_75       | bet_to = to_call + 0.75 * (pot + to_call)  |
| 6   | RAISE_100      | pot-sized raise after call                 |
| 7   | RAISE_150      | 1.5× pot                                   |
| 8   | RAISE_200      | 2.0× pot                                   |
| 9   | RAISE_300      | 3.0× pot                                   |
| 10  | ALL_IN         | shove effective stack                      |

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

## Current-state vector `x` (812 floats)

Three regions: **static state** (121), **legal-actions mask** (11),
**fixed-position action history** (680 = 34 × 20).

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

### Fixed-position action history — offsets 132..811 (34 rows × 20 feat = 680 floats)

The history block is partitioned into four contiguous per-street sub-blocks:

| street  | slot count | start offset | end offset (exclusive) |
|---------|-----------:|-------------:|-----------------------:|
| preflop |         10 |          132 |                    332 |
| flop    |          8 |          332 |                    492 |
| turn    |          8 |          492 |                    652 |
| river   |          8 |          652 |                    812 |

Within a sub-block, slot `k` is filled by the `k`th action of that street
(oldest first, slot 0 = first action of the street). Slot positions ARE
deterministic — slot 3 of the flop block is always "the 4th flop action,"
never anything else.

If a street has fewer actions than its slot budget, the trailing slots are
zero-padded; their `is_real` bit (offset 0 within the row) is 0.

If a street has *more* actions than its slot budget, `encode()` throws
`std::runtime_error` rather than silently truncating. Budgets were sized to
cover the empirical worst case under uniform-random play (100k hands → max
`[9, 7, 7, 6]`); trained policies pick larger raises and produce shorter
sequences, so the throw should never fire in practice.

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

Removed vs v0.3: per-row street one-hot (slot implies street under fixed
slots) and `pos_norm` (slot implies position within street).

Note: the actor one-hot is resolved **at observation time** — `was_us` is
"were *you* the actor", i.e. `act.actor == state.to_act`. When the hand ends,
`encode()` refuses to run (terminal states are never observed).

**Total: 121 + 11 + 680 = 812 floats.** The offsets above are **load-bearing**.

## Tensor shapes (Python side)

```python
obs.x         : np.ndarray[shape=(812,),        dtype=float32]
obs.a         : np.ndarray[shape=(n_legal, 11), dtype=float32]
obs.legal     : list[ActionType] of length n_legal
obs.legal_idx : np.ndarray[shape=(n_legal,),    dtype=int8]
```

Module-level constants (`pokertrainer_engine`):

```
X_DIM, A_DIM, HIST_MAX, HIST_FEAT, STATIC_DIM, LEGAL_MASK_DIM, NUM_ACTIONS,
X_OFF_LEGAL_MASK, X_OFF_PREFLOP, X_OFF_FLOP, X_OFF_TURN, X_OFF_RIVER,
PREFLOP_SLOTS, FLOP_SLOTS, TURN_SLOTS, RIVER_SLOTS,
STREET_SLOTS = (10, 8, 8, 8),
STREET_OFFSETS = (X_OFF_PREFLOP, X_OFF_FLOP, X_OFF_TURN, X_OFF_RIVER),
```

C ABI exposes the same via `pt_x_dim()` / `pt_hist_max()` / `pt_hist_feat()` /
`pt_x_off_street(street)` / `pt_street_slots(street)`.
