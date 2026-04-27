set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc \
    --device cuda:0 \
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
    --ckpt-dir runs/gpu_1M_clip10
