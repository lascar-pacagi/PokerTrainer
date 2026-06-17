# Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ┌─────────────────┐     ┌────────────────────┐    ┌────────────────┐   │
│   │   Flutter UI    │     │   Python Trainer   │    │  Rust Validator │  │
│   │   (desktop)     │     │   (PyTorch, GPU)   │    │ (postflop-solver)│  │
│   └────────┬────────┘     └──────────┬─────────┘    └────────┬────────┘   │
│            │ dart:ffi                │ pybind11              │ bin exec   │
│            │                         │                       │            │
│            ▼                         ▼                       │            │
│   ┌─────────────────────────────────────────────┐            │            │
│   │             C++ Engine (libpokertrainer.so) │            │            │
│   │  ┌──────┐  ┌────────┐  ┌────────┐  ┌─────┐  │            │            │
│   │  │Cards │→ │HandEval│  │GameHU  │→ │Env  │  │            │            │
│   │  └──────┘  └────────┘  └───┬────┘  └──┬──┘  │            │            │
│   │            ┌────────┐      │          │     │            │            │
│   │            │Encoder │←─────┘          │     │            │            │
│   │            └───┬────┘                 │     │            │            │
│   │                ▼                      ▼     │            │            │
│   │         ┌──────────────┐   ┌────────────┐   │            │            │
│   │         │  LibTorch    │   │ C ABI for  │   │            │            │
│   │         │  inference   │   │ Flutter    │   │            │            │
│   │         └──────────────┘   └────────────┘   │            │            │
│   └─────────────────────────────────────────────┘            │            │
│                            ▲                                 │            │
│                            │  TorchScript (.pt)              │            │
│                            │                                 │            │
│   ┌──────────────────────────────────┐                       │            │
│   │  Training checkpoints + strategy │───────────────────────┘            │
│   │  JSON exports (trainer/runs/)    │                                    │
│   └──────────────────────────────────┘                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Cross-language boundaries

### Python ↔ C++ (training time)

- `pybind11` module `pokertrainer_engine` wraps `Env`.
- Actor processes each own a `pokertrainer_engine.Env`; they call `reset()`,
  `step(action_idx)`, and `observation()` to advance games.
- The observation is an `EncodedState` exposing NumPy arrays (`x`, `a`,
  `legal`, `legal_idx`) — zero-copy where possible via `py::array_t`. There is
  no separate `z` history tensor: history is folded into `x` (and, for the CFR
  trainers, into a token sequence — see `trainer/cfr/tokenize.py`).
- The learner stays pure Python + PyTorch. The trained model is the only
  artifact that crosses back to C++ (TorchScript export).

### Flutter ↔ C++ (play time)

- Engine exposes a plain-C ABI in `engine/bindings/ffi.cpp`:
  ```c
  void*    pt_env_create(uint64_t seed, int64_t starting_stack_chips);
  void     pt_env_destroy(void*);
  int32_t  pt_env_reset_seed(void*, uint64_t hand_seed);
  int32_t  pt_env_step(void*, int32_t legal_idx, double* reward_out);
  int32_t  pt_env_step_action(void*, int32_t action_type, double* reward_out);
  int32_t  pt_env_step_raise_to(void*, int64_t bet_to_chips, double* reward_out);
  int32_t  pt_env_observe(void*, /* x, a, legal buffers */ ...);
  int32_t  pt_env_legal_sizings(void*, int64_t* bet_to_out);  // chips per legal action
  void*    pt_model_load(const char* torchscript_path);
  int32_t  pt_model_query(void*, /* encoded state -> per-action values */ ...);
  // ... + pt_env_state_snapshot / card + history accessors for UI rendering
  ```
- Dart binds via `dart:ffi` directly (no Platform Channels, no JSON).

### Rust ↔ engine (validation)

- `validation/src/main.rs` is a Rust JSON bridge over `postflop-solver`
  (binary `pt-solver`): it **solves** a postflop subgame and dumps the strategy
  + EV tree as JSON. `--mode tree` (default) serializes every action node up to
  the next chance deal; `--mode root` emits just the root node.
- The C++ validator (`engine/validation/`: `scenario.h` + `best_response.h`)
  loads a `--mode tree` dump and **independently** recomputes best-response /
  exploitability to cross-check the solver. `trainer/validation/solver_distance.py`
  uses `--mode root` for one-shot exploitability checks; the Flutter training
  tab renders `--mode tree`.

## Data flow — one training step

1. Actor `Env.reset()` → deals cards, posts blinds, determines first-to-act.
2. Actor calls `obs = env.observation()` → `(x, a[legal], legal_idx)`.
3. Actor sends `(a, x)` tensors to the shared-memory model for scoring.
4. Model returns `values[n_legal]`; actor picks argmax (or ε-random).
5. Actor calls `env.step(action_idx)` → reward only at terminal; buffers
   accumulate `(x, a_chosen, reward_to_go)`.
6. At game end, Monte-Carlo return is back-propagated as the training target.
7. Learner samples batches, performs gradient step, broadcasts weights via
   shared memory to actor models.

## Data flow — one play-time decision

1. Flutter sends the current `GameHU` state over FFI.
2. Engine builds `(x, a, legal, legal_idx)` via `Encoder`.
3. `Inference::forward()` runs the TorchScript model over `(n_legal, feats)`.
4. Engine returns action index + optional value/EV estimate.
5. Flutter renders the bot's recommended action + the user's action; computes
   and persists the EV delta if the user plays a different action.
