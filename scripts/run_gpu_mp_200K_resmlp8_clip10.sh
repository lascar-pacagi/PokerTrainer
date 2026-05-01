#!/usr/bin/env bash
# Multi-actor (CPU) + GPU learner variant of run_gpu_200K_resmlp8_clip10.sh.
# Same hyperparameters; the only difference is the trainer entry point and
# the actor pool that drives rollouts in parallel.
#
# Tune --mp-actors to your node's core count: ~physical-cores/2 is a
# reasonable starting point. Each actor is single-threaded CPU so they
# scale roughly linearly until the learner becomes the bottleneck (look at
# qsize in the log — if it's growing without bound, actors are out-pacing
# the GPU; if it stays at 0 most of the time, the GPU is starved and you
# can drop --mp-actors).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=engine/build:trainer python -m dmc.dmc_mp_gpu \
    --learner-device cuda:0 \
    --mp-actors 8 \
    --actor-sync-every 16 \
    --queue-maxsize 8192 \
    --drain-per-step 64 \
    --arch resmlp_v1 \
    --mlp-layers 8 \
    --mlp-hidden 512 \
    --mlp-expansion 4 \
    --max-steps 200000 \
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
    --ckpt-dir runs/gpu_mp_200K_resmlp8_clip10
