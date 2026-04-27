#!/usr/bin/env bash
# A/B variant: very tight MC return clip (±10 BB). The aggressive end of
# the clip sweep. Watch for: loss converging fast (<200), but check that
# Q-values don't compress into a tiny range — if mean_pred and mean_tgt
# are both within ±2 BB on every log line, the network may have lost
# the ability to differentiate "value bet for stacks" from "block bet".
# Below ±10 the action ordering would start to wobble (small EV deltas
# get clipped against the boundary).
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
    --reward-clip 10.0 \
    --slot-log-every 500 \
    --eval-every-steps 1000 \
    --ckpt-dir runs/gpu_50k_clip10
