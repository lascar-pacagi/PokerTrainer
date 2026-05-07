#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
    --arch resmlp_v1 \
    --mlp-layers 8 \
    --mlp-hidden 512 \
    --mlp-expansion 4 \
    --max-steps 200000 \
    --rollout-hands 256 \
    --grad-steps-per-iter 64 \
    --buffer-capacity 500000 \
    --min-buffer 20000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 50000 \
    --reward-clip 10.0 \
    --slot-log-every 5000 \
    --eval-every-steps 10000 \
    --checkpoint-every-steps 50000 \
    --ckpt-dir runs/gpu_200K_resmlp8_clip10

