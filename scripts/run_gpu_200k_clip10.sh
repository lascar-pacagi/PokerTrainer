#!/usr/bin/env bash
# 200k clip=10 run with extended exploration warmup (20k decay vs 10k baseline).
# Hypothesis: more exploration during the early plastic phase populates the
# buffer with a wider state distribution, giving the network a chance to
# learn card-conditioning across more hand archetypes before the policy
# greedifies. Pairs naturally with longer total training (200k) so the
# refinement phase has time to consolidate.
#
# Schedule:
#   step 0      eps=1.00  (pure random — every action equally likely)
#   step 5k     eps≈0.78
#   step 10k    eps≈0.55
#   step 15k    eps≈0.33
#   step 20k    eps=0.10  (held flat for the remaining 180k)
#
# Estimated wall: ~30 min on cuda:0.
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
    --epsilon-decay-steps 20000 \
    --reward-clip 10.0 \
    --slot-log-every 2000 \
    --eval-every-steps 5000 \
    --ckpt-dir runs/gpu_200k_clip10_exp20k
