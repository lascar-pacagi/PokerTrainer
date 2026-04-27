# State Encoding Specification

**Status:** v0.3 (flat-sequence layout, heads-up NLHE). Bit-identical outputs
required from `engine/src/encoder.cpp` and `trainer/dmc/features.py`. Any
change here requires updating both and running the parity tests in
`engine/tests/test_encoder.cpp` and `trainer/tests/test_features_parity.py`.

## What changed from v0.2

- **`a` rows are pure 11-dim action one-hots.** All 6 state-dependent
  scalars (`bet_to_bb`, `bet_frac_pot`, `pot_after_bb`, `stack_after_bb`,
  `is_all_in`, `is_raise`) are removed. Action representation no longer
  varies by state — same action ⇒ same `a` row regardless of context.
  `A_DIM` 17 → 11.
- **State-level legal-actions mask added to `x`**: 11 bits at offset 121
  showing which `ActionType` slots are legal in the current state.
- **`is_real` inlined as bit 0 of every history row.** The previous
  external 24-bit valid-mask block (positions 721–744) is removed; padded
  rows distinguish themselves by being all-zero.
- **History rows now populate `stack_after_bb` and `was_all_in` for
  real.** Previously hard-zeroed (engine couldn't reconstruct without
  full replay). `AppliedAction` was extended with `stack_after_chips`
  and `was_all_in` fields, populated during `HUGame::step()`.
- `STATIC_DIM` 121 → 132 (added 11-bit legal mask). `X_DIM` 745 → 732.
  `HIST_FEAT` stays 25 (same width, different content).
- **Rationale for 11-dim one-hot `a`**: the v0.2 design embedded
  state-derived scalars in each row, which (a) made the same action have
  different `a` vectors across states (the network had to *learn* the
  equivalence class), (b) double-counted information already in `x`, and
  (c) routed gradients for shared "what does this action mean" knowledge
  through both the one-hot and scalar weights, smearing the
  representation. Pure one-hots eliminate this.
- **Rationale for the legal mask in `x`**: replaces the implicit
  legality signal that v0.2 carried via `a`'s row count. Gives the
  network a single, well-defined place to read "what is the action menu
  in this state" — useful context for valuing any individual action
  (e.g., a 75% raise means something different when ALL_IN is also
  available vs. when it isn't).

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

## Current-state vector `x` (732 floats)

Three regions: **static state** (121), **legal-actions mask** (11),
**flat action history** (600 = 24 × 25). No external valid-mask block.

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

### Flat action history — offsets 132..731 (24 rows × 25 feat = 600 floats)

Rows written **oldest-first**. Row `k` occupies offsets `132 + k*25 ..
132 + (k+1)*25 - 1`. If the hand has fewer than 24 actions so far, unused
rows are zero-padded — their `is_real` bit (offset 0 within the row) is 0.

Each row's 25 floats:

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
| 20–23             |  4   | street one-hot {preflop, flop, turn, river} at time of action |
| 24                |  1   | pos_norm = k / 24 (row index normalized to [0, 1))     |

Note: the actor one-hot is resolved **at observation time** — `was_us` is
"were *you* the actor", i.e. `act.actor == state.to_act`. When the hand ends,
`encode()` refuses to run (terminal states are never observed).

**Total: 121 + 11 + 600 = 732 floats.** The offsets above are **load-bearing**.

## Tensor shapes (Python side)

```python
obs.x         : np.ndarray[shape=(732,),        dtype=float32]
obs.a         : np.ndarray[shape=(n_legal, 11), dtype=float32]
obs.legal     : list[ActionType] of length n_legal
obs.legal_idx : np.ndarray[shape=(n_legal,),    dtype=int8]
```

There is no `obs.z` and no separate valid-mask block.

## Parity test requirement

`engine/tests/test_encoder.cpp` and `trainer/tests/test_features_parity.py`
must independently produce byte-identical `x` and `a` for a fixed corpus of
hand-rolled fixtures covering:

1. Dealt preflop, OOP to act, no prior action.
2. 3-bet preflop spot, IP to act.
3. Flopped set on a wet board, OOP c-bet + raise + reraise sequence.
4. Turn all-in shove by IP.
5. River check-check going to showdown.

Fixtures stored as JSON in `engine/tests/fixtures/encoder/*.json`; both
languages load and compare byte-for-byte.
