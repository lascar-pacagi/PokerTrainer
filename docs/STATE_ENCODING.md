# State Encoding Specification

**Status:** v0.2 (flat-sequence layout, heads-up NLHE). Bit-identical outputs
required from `engine/src/encoder.cpp` and `trainer/dmc/features.py`. Any
change here requires updating both and running the parity tests in
`engine/tests/test_encoder.cpp` and `trainer/tests/test_features_parity.py`.

## What changed from v0.1 / v0.2

- **Dropped the `z` history tensor and the LSTM.** The full action history is
  now flattened into `x` as 24 fixed-width rows + a 24-bit valid-mask.
- **Added `RAISE_25` (0.25× pot) as the smallest raise slot** — supports block
  bets and tiny c-bets that the previous 0.33-floor couldn't express.
  `NUM_ACTIONS` 10 → 11, `A_DIM` 16 → 17, `HIST_FEAT` 24 → 25, `X_DIM` 721 → 745.
- Rationale: HU NLHE has a tiny action alphabet (11 types) and a bounded
  sequence (≤ 24 actions/hand). Flattening gives the MLP direct access to every
  past action without a recurrent bottleneck; the per-slot weights specialize
  to "row k means action-index-k." LSTM added training instability on MC
  targets without obvious representational gain for a 24-step bounded sequence.

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
RAISE_33 — that's expected, the network learns to ignore duplicate slots.

## Per-action encoding vector `a` (17 floats)

| slice          | dims | meaning                                               |
|----------------|------|-------------------------------------------------------|
| action_type    | 11   | one-hot over {fold, check/call, 8× raise, all-in}     |
| bet_to_bb      |  1   | final bet-to amount in bb (0 for fold)                |
| bet_frac_pot   |  1   | final bet-to as fraction of pre-bet pot (0 for fold)  |
| pot_after_bb   |  1   | pot in bb *after* action                              |
| stack_after_bb |  1   | our stack in bb *after* action                        |
| is_all_in      |  1   | 1 if action would put us all-in                       |
| is_raise       |  1   | 1 if action is a raise                                |

Total: **17 floats per legal action**. Scoring a decision builds a
`(n_legal, 17)` tensor and the network processes each row alongside the same
`x`.

## Current-state vector `x` (745 floats)

The vector has three regions: **static state** (121), **flat action history**
(600 = 24 × 25), **history valid-mask** (24).

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

### Flat action history — offsets 121..720 (24 rows × 25 feat = 600 floats)

Rows written **oldest-first**. Row `k` occupies offsets `121 + k*25 ..
121 + (k+1)*25 - 1`. If the hand has fewer than 24 actions so far, unused rows
are zero-padded and their valid-mask bit is 0.

Each row's 25 floats:

| within-row offset | dims | meaning                                                |
|-------------------|------|--------------------------------------------------------|
| 0–10              | 11   | action_type one-hot (same indexing as `a`)             |
| 11                |  1   | bet_to_bb                                              |
| 12                |  1   | bet_frac_pot (0 for fold/check)                        |
| 13                |  1   | pot_after_bb                                           |
| 14                |  1   | stack_after_bb (= 0 in history: actor's post-action stack is not tracked here) |
| 15                |  1   | is_all_in (history: always 0; kept for layout parity)  |
| 16                |  1   | is_raise                                               |
| 17                |  1   | actor_was_us (1 if actor == current to_act player)     |
| 18                |  1   | actor_was_villain (1 otherwise)                        |
| 19–22             |  4   | street one-hot {preflop, flop, turn, river} at time of action |
| 23                |  1   | pos_norm = k / 24 (row index normalized to [0, 1))     |
| 24                |  1   | pad (reserved)                                         |

Note: the actor one-hot is resolved **at observation time** — `was_us` is
"were *you* the actor", i.e. `act.actor == state.to_act`. When the hand ends,
`encode()` refuses to run (terminal states are never observed).

### Valid-mask — offsets 721..744 (24 floats)

`x[721 + k] = 1.0` if row `k` of the history contains a real action,
else `0.0`. The MLP can use this to ignore zero-padded rows.

**Total: 121 + 600 + 24 = 745 floats.** The offsets above are **load-bearing**.

## Tensor shapes (Python side)

```python
obs.x         : np.ndarray[shape=(745,),        dtype=float32]
obs.a         : np.ndarray[shape=(n_legal, 17), dtype=float32]
obs.legal     : list[ActionType] of length n_legal
obs.legal_idx : np.ndarray[shape=(n_legal,),    dtype=int8]
```

There is no `obs.z`.

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
