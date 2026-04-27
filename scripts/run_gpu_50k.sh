#!/usr/bin/env bash
# 50k single-process GPU run. Single learner + single actor (sequential
# rollout/learn) on cuda:0. Faster per-step than mp+CPU because the bottleneck
# is the gradient step on a 1.4M-param MLP, not the rollout.
#
# Pairs with the encoder fix from 2026-04-25 (project_pybind_array_bug.md) and
# the post-fix v2 anti-collapse defaults: pure-random eps warmup, decay over
# 10k steps, ±25 BB MC return clip.
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
    --reward-clip 25.0 \
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --ckpt-dir runs/gpu_50k
