#!/usr/bin/env bash
# Anti-collapse 50k mp run: bumped eps floor, longer decay, MC return clip,
# buffer-side slot telemetry every 500 steps. See project_design_decisions.md
# entries 4/10/11 for rationale.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc_mp \
    --mp-actors 4 \
    --max-steps 50000 \
    --grad-steps-per-iter 64 \
    --drain-per-step 128 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 10000 \
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --checkpoint-every-steps 1000 \
    --ckpt-dir runs/cpu_mp_50k_v2
