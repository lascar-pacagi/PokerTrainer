# PokerTrainer

Heads-up No-Limit Hold'em trainer powered by a DouZero-style Deep Monte-Carlo
(DMC) agent, with a Flutter UI that surfaces the GTO-correct action at every
decision and tracks your progress over time.

**Status:** Phase 0 (engine foundation). Under construction.

## Layout

```
engine/      # C++20 — rules, hand evaluator, state encoder, LibTorch inference
trainer/     # Python 3.11 + PyTorch — DMC actor-learner, adapted from DouZero
validation/  # Rust — exploitability check via postflop-solver
ui/          # Flutter — desktop-first, dart:ffi to engine, SQLite persistence
docs/        # ARCHITECTURE.md, STATE_ENCODING.md, TRAINING_RECIPE.md
```

## Phases

| # | Goal | Status |
|---|------|--------|
| 0 | C++ engine + pybind11 bindings (HU NLHE, state encoder, hand eval) | in progress |
| 1 | DMC training loop, single-GPU then multi-GPU | pending |
| 2 | Validation vs. postflop-solver, < 50 mbb/hand target | pending |
| 3 | Flutter UI with coach overlay + session stats + hand replay | pending |
| 4 | 6-max extension | future |

## Quickstart (once scaffolded)

```bash
# Engine
cd engine && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build

# Trainer
cd trainer && pip install -e . && python -m dmc.dmc --smoke

# Validation
cd validation && cargo run --release -- --ckpt ../trainer/runs/latest --n-spots 1000

# UI
cd ui && flutter run -d linux
```

## Design docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, data flow, cross-language boundaries
- [docs/STATE_ENCODING.md](docs/STATE_ENCODING.md) — the tensor layout contract (must stay in lockstep between C++ encoder and Python features)
- [docs/TRAINING_RECIPE.md](docs/TRAINING_RECIPE.md) — hparams, wall-clock, checkpoints
