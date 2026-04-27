#!/usr/bin/env bash
# Mirror of run_gpu_1M_clip10.sh, but with the resmlp_v1 architecture:
# 8 pre-LN residual blocks, hidden=512, expansion=4 → ~17.2M params (vs 1.44M).
# All other training hparams kept identical for clean A/B comparison.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
    --arch resmlp_v1 \
    --mlp-layers 8 \
    --mlp-hidden 512 \
    --mlp-expansion 4 \
    --max-steps 1000000 \
    --rollout-hands 256 \
    --grad-steps-per-iter 64 \
    --buffer-capacity 500000 \
    --min-buffer 10000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.10 \
    --epsilon-decay-steps 50000 \
    --reward-clip 10.0 \
    --slot-log-every 5000 \
    --eval-every-steps 10000 \
    --checkpoint-every-steps 50000 \
    --ckpt-dir runs/gpu_1M_resmlp8_clip10

