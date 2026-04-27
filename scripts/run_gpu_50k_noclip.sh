#!/usr/bin/env bash
# A/B variant of run_gpu_50k.sh with NO reward clip. Direct apples-to-apples
# with the mp+CPU no-clip run (which scored +603k vs station, +40k vs
# check_fold). Confirms whether the clip caused the vs-station drop or
# something else (GPU vs CPU, single-process vs mp).
#
# Expect: large grad norms (~70k clip-saturated), bouncy loss (~thousands).
# Eval slot mix should resemble the mp run more than the clip=25 run.
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
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --ckpt-dir runs/gpu_50k_noclip
