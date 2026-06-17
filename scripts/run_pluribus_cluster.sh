#!/usr/bin/env bash
# Cluster launch wrapper for the Pluribus BLUEPRINT trainer
# (pluribus.blueprint_mp) inside the Singularity image — a CPU actor pool with
# additive table merge. NO GPU: Pluribus is tabular (regret tables over an
# abstracted game), so the blueprint is CPU/RAM-bound, not FLOP-bound. This is
# the 64-core / big-RAM profile Pluribus famously trained on (no --nv needed).
#
# Usage:
#   bash scripts/run_pluribus_cluster.sh /path/to/pokertrainer.sif /scratch/$USER/pluribus_run
#
# Override defaults via env:
#   WORKERS=64 STACK_BB=10 PRESET=pushfold ROUNDS=20000 TRAV=512 \
#       bash scripts/run_pluribus_cluster.sh <sif> <ckpt>
#
# Presets:
#   pushfold — short-stack jam/fold blueprint (matches the Nash oracle; cheap).
#   discrete — a per-street pot-fraction grid; postflop needs a bucket cache
#              (BUCKET_CACHE=...) built by learn_street_buckets, else it falls
#              back to a coarse preflop-class abstraction (a warning is printed).
#
# Notes:
#   * ALLIN_SAMPLES (>0) Monte-Carlo all-in equity over that many runouts —
#     large variance reduction at short stacks (a called all-in swings ±stack).
#   * The blueprint trains with NO neural net and NO GPU; the image only needs
#     the built engine (pokertrainer_engine.so) + numpy. The same cfr.def image
#     works (its torch/CUDA layers are simply unused here).
set -euo pipefail

SIF="${1:?usage: run_pluribus_cluster.sh <path-to-sif> <ckpt-dir>}"
CKPT_DIR="${2:?ckpt-dir required (e.g. /scratch/$USER/pluribus_run)}"

WORKERS="${WORKERS:-$(nproc)}"
ROUNDS="${ROUNDS:-20000}"
TRAV="${TRAV:-512}"                 # traversals per worker per round
STACK_BB="${STACK_BB:-10}"
PRESET="${PRESET:-pushfold}"
ALLIN_SAMPLES="${ALLIN_SAMPLES:-8}"
BUCKET_CACHE="${BUCKET_CACHE:-}"
CKPT_EVERY="${CKPT_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-20}"
SEED="${SEED:-0}"
START_METHOD="${START_METHOD:-fork}"

mkdir -p "$CKPT_DIR"
CKPT_DIR="$(readlink -f "$CKPT_DIR")"

# No --nv: the blueprint is CPU-only. Bind the checkpoint dir for output.
singularity exec \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    bash -lc "cd /opt/pokertrainer 2>/dev/null || cd \$PWD; \
        PYTHONPATH=engine/build:trainer python -u -m pluribus.blueprint_mp \
        --workers '$WORKERS' \
        --rounds '$ROUNDS' \
        --traversals-per-worker '$TRAV' \
        --stack-bb '$STACK_BB' \
        --action-preset '$PRESET' \
        --allin-equity-samples '$ALLIN_SAMPLES' \
        ${BUCKET_CACHE:+--bucket-cache '$BUCKET_CACHE'} \
        --ckpt-dir '$CKPT_DIR' \
        --ckpt-every '$CKPT_EVERY' \
        --log-every '$LOG_EVERY' \
        --start-method '$START_METHOD' \
        --seed '$SEED'"
