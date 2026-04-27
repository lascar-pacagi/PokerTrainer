#!/usr/bin/env bash
# Canonical long run after the clip sweep: 100k steps at the proven-best
# constant clip=10. Baseline question — does clip=10 plateau at ~+460k vs
# station, or does longer training keep improving?
#
# Reference (project_reward_clip_sweep.md): clip=10 at 50k won the sweep on
# loss (55), grad (251), vs check_fold (+42.8k), and vs station (+460.2k).
# The schedule variant (start=20, end=80) underperformed because late-phase
# loose clip overwrites early aggressive learning — see same memory entry.
#
# Estimated wall: ~15 min on cuda:0.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
    --max-steps 100000 \
    --rollout-hands 256 \
    --grad-steps-per-iter 64 \
    --buffer-capacity 200000 \
    --min-buffer 4000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 10000 \
    --reward-clip 10.0 \
    --slot-log-every 1000 \
    --eval-every-steps 2000 \
    --ckpt-dir runs/gpu_100k_clip10
