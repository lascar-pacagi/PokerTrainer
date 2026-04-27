#!/usr/bin/env bash
# Diagnostic 2 of 2: 100k steps, EXTENDED 20k epsilon decay (the warmup of
# the regressed 200k_exp20k run). Tests "is the warmup itself harmful at
# the same training duration?" If this run regresses vs 100k_clip10
# (eps_decay=10k) → the long warmup hurt independent of training length.
# If this matches 100k_clip10 → warmup is harmless and the 200k regression
# was purely about extra training time.
#
# Pair with run_gpu_200k_clip10_exp10k.sh to complete the 2×2 design.
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
    --epsilon-decay-steps 20000 \
    --reward-clip 10.0 \
    --slot-log-every 1000 \
    --eval-every-steps 2000 \
    --ckpt-dir runs/gpu_100k_clip10_exp20k
