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
- The observation is a dict of NumPy arrays (`x`, `a`, `z`, `legal_idx`) —
  zero-copy where possible via `py::array_t`.
- The learner stays pure Python + PyTorch. The trained model is the only
  artifact that crosses back to C++ (TorchScript export).

### Flutter ↔ C++ (play time)

- Engine exposes a plain-C ABI in `engine/bindings/ffi.cpp`:
  ```c
  void*    pt_env_new(int seed);
  void     pt_env_free(void*);
  void     pt_env_reset(void*);
  int32_t  pt_env_step(void*, int32_t action_idx);
  int32_t  pt_env_legal_mask(void*, uint8_t* out);      // 10-byte buffer
  int32_t  pt_env_recommend(void*);                     // uses loaded model
  void     pt_load_model(const char* torchscript_path);
  // ... observation accessors for UI rendering
  ```
- Dart binds via `dart:ffi` directly (no Platform Channels, no JSON).

### Rust ↔ engine (validation)

- Validator reads an **exported strategy JSON** produced by the trainer
  (`trainer/export/strategy.py`) for a specific subgame. It does *not* link
  to the C++ engine at runtime — it just needs the strategy as frequencies
  per infoset.
- Runs `postflop-solver` to best-respond against it and reports exploitability.

## Data flow — one training step

1. Actor `Env.reset()` → deals cards, posts blinds, determines first-to-act.
2. Actor calls `obs = env.observation()` → `(x, a[legal], z, legal_idx)`.
3. Actor sends `(a, x, z)` tensors to the shared-memory model for scoring.
4. Model returns `values[n_legal]`; actor picks argmax (or ε-random).
5. Actor calls `env.step(action_idx)` → reward only at terminal; buffers
   accumulate `(x, a_chosen, z, reward_to_go)`.
6. At game end, Monte-Carlo return is back-propagated as the training target.
7. Learner samples batches, performs gradient step, broadcasts weights via
   shared memory to actor models.

## Data flow — one play-time decision

1. Flutter sends the current `GameHU` state over FFI.
2. Engine builds `(x, a, z, legal_idx)` via `Encoder`.
3. `Inference::forward()` runs the TorchScript model over `(n_legal, feats)`.
4. Engine returns action index + optional value/EV estimate.
5. Flutter renders the bot's recommended action + the user's action; computes
   and persists the EV delta if the user plays a different action.
