#!/usr/bin/env bash
# 100k GPU run with a TIGHT→LOOSE reward-clip schedule.
#
# Rationale (from project_reward_clip_sweep.md): tight clip teaches the
# network aggressive card-conditional play (low-variance gradient signal,
# bounded losses on aggressive actions). Once that's locked in, loosening
# the clip lets the network see real upside variance — important for
# learning to extract stacks from sticky opponents (the regime where
# clip=25 left value on the table vs calling_station).
#
# Schedule: clip=20 → clip=80 over the first 70k of 100k steps. Last 30k
# stays at clip=80, near-uncapped at 100bb stacks. Pair with the proven
# eps_start=1.0 → 0.10 warmup over 10k steps.
#
# Estimated wall: ~15 min on cuda:0 (~110 steps/sec).
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
    --reward-clip 20.0 \
    --reward-clip-end 80.0 \
    --reward-clip-decay-steps 70000 \
    --slot-log-every 1000 \
    --eval-every-steps 2000 \
    --ckpt-dir runs/gpu_100k_clip_sched
