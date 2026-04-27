#!/usr/bin/env bash
# A/B variant: tighter MC return clip (±15 BB). Hypothesis from the
# clip=25 vs clip=50 result: tighter clip → more aggressive policy
# (because bounded losses make non-fold actions look less catastrophic
# relative to fold's -0.5 BB). clip=25 was the Goldilocks of the first
# three; pushing tighter to ±15 should preserve or improve vs-station
# / vs-check_fold without driving mode collapse.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
    --max-steps 50000 \
    --rollout-hands 256 \
    --grad-steps-per-iter 64 \
    --buffer-capacity 200000 \
    --min-buffer 4000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 10000 \
    --reward-clip 15.0 \
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --ckpt-dir runs/gpu_50k_clip15
