#!/usr/bin/env bash
# Cluster launch wrapper for ReBeL training (rebel_py.rebel_mp_gpu) inside the
# Singularity image — CPU actor pool + GPU learner (mirrors run_cfr_cluster.sh).
#
# Usage:
#   bash scripts/run_rebel_cluster.sh /path/to/pokertrainer.sif /scratch/$USER/rebel_run
#
# Override defaults via env:
#   GAME=turn ACTORS=32 STACK_BB=6 CFR_ITERS=512 N_BOARDS=8 EPOCHS=400 \
#       bash scripts/run_rebel_cluster.sh <sif> <ckpt>
#
# Games:
#   river — Phase 2a (betting-depth limit, no chance): cheap, light memory.
#   turn  — Phase 2b (turn→river, one chance node): the cross-street value net.
#           Each turn tree holds 48 per-card showdown matrices (~0.3 GB), so a
#           turn actor needs ~N_BOARDS*0.3 GB RAM — size ACTORS/N_BOARDS to fit.
#
# Notes:
#   * CFR_ITERS sets the *accuracy of the value targets* (the time-averaged root
#     value has O(1/T) bias the net learns) — keep it generous (256+).
#   * The image needs torch (CUDA) + the built engine (pokertrainer_engine.so).
set -euo pipefail

SIF="${1:?usage: run_rebel_cluster.sh <path-to-sif> <ckpt-dir>}"
CKPT_DIR="${2:?ckpt-dir required (e.g. /scratch/$USER/rebel_run)}"

GAME="${GAME:-turn}"
ACTORS="${ACTORS:-32}"
N_BOARDS="${N_BOARDS:-4}"
STACK_BB="${STACK_BB:-6}"
CFR_ITERS="${CFR_ITERS:-256}"
RAND_ACTION_PROB="${RAND_ACTION_PROB:-0.25}"
EPOCHS="${EPOCHS:-300}"
EXAMPLES_PER_EPOCH="${EXAMPLES_PER_EPOCH:-8000}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-128}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BUFFER_CAP="${BUFFER_CAP:-2000000}"
N_HIDDEN="${N_HIDDEN:-512}"
N_LAYERS="${N_LAYERS:-3}"
LR="${LR:-5e-4}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EVAL_ITERS="${EVAL_ITERS:-256}"
CKPT_EVERY="${CKPT_EVERY:-20}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

mkdir -p "$CKPT_DIR"
CKPT_DIR="$(readlink -f "$CKPT_DIR")"

# PYTHONPATH inside the image: the built engine + the trainer package. Adjust
# the bind/paths to match how the .sif was built (see scripts/cfr.def).
singularity exec --nv \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    bash -lc "cd /opt/pokertrainer 2>/dev/null || cd \$PWD; \
        PYTHONPATH=engine/build:trainer python -u -m rebel_py.rebel_mp_gpu \
        --game '$GAME' \
        --mp-actors '$ACTORS' \
        --n-boards '$N_BOARDS' \
        --stack-bb '$STACK_BB' \
        --cfr-iters '$CFR_ITERS' \
        --random-action-prob '$RAND_ACTION_PROB' \
        --n-epochs '$EPOCHS' \
        --examples-per-epoch '$EXAMPLES_PER_EPOCH' \
        --batches-per-epoch '$BATCHES_PER_EPOCH' \
        --batch-size '$BATCH_SIZE' \
        --buffer-cap '$BUFFER_CAP' \
        --n-hidden '$N_HIDDEN' \
        --n-layers '$N_LAYERS' \
        --lr '$LR' \
        --eval-every '$EVAL_EVERY' \
        --eval-iters '$EVAL_ITERS' \
        --checkpoint-every '$CKPT_EVERY' \
        --learner-device '$DEVICE' \
        --seed '$SEED' \
        --ckpt-dir '$CKPT_DIR'"
