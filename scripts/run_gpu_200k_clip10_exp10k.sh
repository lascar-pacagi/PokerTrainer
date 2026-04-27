#!/usr/bin/env bash
# Diagnostic 1 of 2: 200k steps, SAME 10k epsilon decay as the 100k_clip10
# baseline. Tests "is the regression at 200k_exp20k from more training, or
# from the extended warmup?" If this run also regresses → more training
# crystallizes the policy. If this matches/beats 100k_clip10 → the
# extended warmup was the issue.
#
# Pair with run_gpu_100k_clip10_exp20k.sh to complete the 2×2 design.
#
# Estimated wall: ~25 min on cuda:0.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
    --max-steps 200000 \
    --rollout-hands 256 \
    --grad-steps-per-iter 64 \
    --buffer-capacity 200000 \
    --min-buffer 4000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 10000 \
    --reward-clip 10.0 \
    --slot-log-every 2000 \
    --eval-every-steps 5000 \
    --ckpt-dir runs/gpu_200k_clip10_exp10k
