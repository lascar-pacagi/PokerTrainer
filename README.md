# PokerTrainer

Heads-up No-Limit Hold'em study tool. Combines a C++ game engine, two
training pipelines (Deep Monte-Carlo and Deep CFR), a Rust GTO solver
bridge, an independent C++ best-response validator, and a Flutter desktop
UI with a GTO-Wizard-style training tab + interactive trainer.

## Layout

```
engine/             # C++20 — rules, hand evaluator, state encoder, FFI ABI
  src/              #   core: card, action, hand_eval, game_hu, encoder, env, inference
  bindings/         #   pybind11 + C ABI for dart:ffi
  validation/       #   independent C++ best-response validator (Catch2 tests)
  data/             #   poker_tables.bin (perfect-hash hand-eval tables)

trainer/            # Python 3.11 + PyTorch
  dmc/              #   Deep Monte-Carlo actor-learner (DouZero-style)
  cfr/              #   Deep CFR variants (single + multi-process)
  evaluate/         #   match-play eval, exploitability sampling
  export/           #   TorchScript serialization for engine inference
  scripts/          #   run helpers + cluster recipes

validation/         # Rust — postflop-solver bridge
  src/main.rs       #   pt-solver CLI: tree-mode JSON dump, dry-run, depth control
  fixtures/         #   sample SpotInput specs

ui/                 # Flutter desktop (Linux/macOS/Windows)
  lib/training/     #   GTO study tab: solver runner, scenario navigator,
                    #     range editor, preflop presets, chance walker,
                    #     trainer mode (test yourself vs solver)
  lib/widgets/      #   inspector tab: table view, action bar, model loader

scripts/            # Cluster + GPU launch scripts
docs/               # ARCHITECTURE.md, STATE_ENCODING.md
validation_runs/    # Solved scenario JSONs (consumed by the UI training tab)
```

## What's working

| Component | Status |
|-----------|--------|
| C++ engine: HU NLHE rules, encoder, FFI, hand-eval | ✅ stable |
| pybind11 bindings for trainer | ✅ stable |
| dart:ffi inspector + per-seat agent dropdown + TorchScript model loader | ✅ stable |
| DMC trainer (single-process + multi-process GPU variant) | ✅ stable |
| Deep CFR trainer (single + multi-process, cluster recipe) | ✅ stable |
| pt-solver Rust bridge (`--mode tree --depth flop\|turn\|river`, `--dry-run`) | ✅ stable |
| Flutter Training tab: 13×13 chart + scenario navigator + tooltips everywhere | ✅ stable |
| Visual range editor (paint + brush + 16 preflop presets) | ✅ stable |
| On-demand subgame re-solving from chance_pending nodes | ✅ stable |
| Trainer mode (interactive): convergence-aware verdicts, EV-tolerant feedback | ✅ stable |
| Independent BR validator (`pt_validate`) — exact + iso-aware + MC sampling | ✅ stable |

## Quickstart

```bash
# ── Engine: build core lib + Catch2 tests + FFI shared lib ─────────
cd engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j && ctest --test-dir build

# ── pt-solver Rust bridge ──────────────────────────────────────────
cd ../validation
cargo build --release
# Solve and dump a sample spot:
./target/release/pt-solver \
    --input fixtures/smoke_srp_flop.json \
    --output ../validation_runs/scenarios/srp.json \
    --mode tree --depth flop

# ── Independent best-response validator (paranoid) ─────────────────
cd ../engine
cmake -S . -B build -DPT_BUILD_VALIDATION=ON
cmake --build build --target pt_validate pt_validation_tests -j
./build/validation/pt_validate --input ../validation_runs/scenarios/srp.json
./build/validation/pt_validation_tests   # 7 cases, 37 assertions

# ── Trainer ────────────────────────────────────────────────────────
cd ../trainer
pip install -e .
python -m dmc.dmc --smoke                      # tiny smoke run
python -m cfr.deep_cfr --smoke                 # Deep CFR smoke

# ── Flutter UI (Linux desktop) ─────────────────────────────────────
cd ../ui
flutter pub get && flutter run -d linux
# Inspector tab: load a TorchScript model, watch policy frame-by-frame.
# Training tab: pick a solved scenario, walk the action tree, paint
# ranges, train against the solver.
```

## Validator precision regimes

The independent BR validator (`engine/validation/pt_validate`) operates in
two modes depending on what's in the scenario JSON:

- **All 5 board cards pinned** (no chance walking) → exact BR. Agreement
  with pt-solver is at floating-point precision (~1µchip on 100-pot).
- **Chance walked** (only flop or only flop+turn pinned) → iso-aware BR.
  Suit-isomorphism handled via independent reconstruction of the swap
  permutation; residual is structural (~0.01–0.03 chips on 100-pot,
  amplified normalization noise compounded through nested chance levels).

Across the test suite (smoke + paranoid: rainbow iso, wide-range,
convergence monotonicity, river-vs-flop cross-check), pt-solver agrees
with the independent BR on every spot tested. No solver bugs found.

## Design docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system diagram, data flow, cross-language boundaries
- [docs/STATE_ENCODING.md](docs/STATE_ENCODING.md) — tensor layout contract between C++ encoder and Python features
