#!/usr/bin/env bash
# A/B variant of run_gpu_50k.sh with a wider reward clip (±50 BB instead of
# ±25 BB). Hypothesis: ±25 truncates the upside the network needs to learn
# stack-extraction shoves vs sticky opponents (calling_station +322k at clip=25
# vs +603k uncached on the mp run). ±50 should keep convergence but recover
# the upside.
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
    --reward-clip 50.0 \
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --ckpt-dir runs/gpu_50k_clip50
