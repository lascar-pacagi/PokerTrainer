#!/usr/bin/env bash
# Cluster launch wrapper for Deep CFR training inside the Singularity image.
#
# Tested on rtx-8000 (sm_75), A40 (sm_86), L40S (sm_89). All three work with
# the same .sif (cu124 wheel falls back to fp32/fp16 on Turing — no bf16, but
# the AdvNet refit is small enough that it doesn't matter).
#
# Usage:
#   bash scripts/run_cfr_cluster.sh /path/to/pokertrainer-cfr.sif /scratch/$USER/cfr_run
#
# Override defaults via env:
#   ACTORS=32 ITERS=500 K=10000 HIDDEN=512 LAYERS=8 \
#       bash scripts/run_cfr_cluster.sh ...
#
# Pre-flight checks expected on the cluster:
#   * a CUDA-visible GPU (nvidia-smi works inside `singularity exec --nv`)
#   * /scratch/$USER writable
#   * --bind /scratch if the image was built without it accessible
set -euo pipefail

SIF="${1:?usage: run_cfr_cluster.sh <path-to-sif> <ckpt-dir>}"
CKPT_DIR="${2:?ckpt-dir required (e.g. /scratch/$USER/cfr_run)}"

ACTORS="${ACTORS:-128}"
ITERS="${ITERS:-1000}"
K="${K:-5000}"
ADV_GRAD_STEPS="${ADV_GRAD_STEPS:-5000}"
POLICY_GRAD_STEPS="${POLICY_GRAD_STEPS:-50000}"
HIDDEN="${HIDDEN:-1024}"
LAYERS="${LAYERS:-10}"
ADV_CAP="${ADV_CAP:-20000000}"
POL_CAP="${POL_CAP:-40000000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

mkdir -p "$CKPT_DIR"
# Singularity's --bind requires an ABSOLUTE destination path. sbatch's
# $PWD-relative paths (e.g. `run`) pass the bash variable check but fail
# inside Singularity with "destination must be an absolute path". Resolve
# now so the rest of the script (and the --ckpt-dir flag) get a path that
# means the same thing from any CWD.
CKPT_DIR="$(readlink -f "$CKPT_DIR")"

echo "[run_cfr_cluster] sif=$SIF"
echo "[run_cfr_cluster] ckpt=$CKPT_DIR"
echo "[run_cfr_cluster] actors=$ACTORS iters=$ITERS K=$K"
echo "[run_cfr_cluster] hidden=$HIDDEN layers=$LAYERS device=$DEVICE"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

singularity exec --nv \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    python3 -m cfr.cfr_mp_gpu \
        --mp-actors           "$ACTORS" \
        --learner-device      "$DEVICE" \
        --n-iterations        "$ITERS" \
        --n-traversals-per-iter "$K" \
        --adv-grad-steps      "$ADV_GRAD_STEPS" \
        --policy-grad-steps   "$POLICY_GRAD_STEPS" \
        --hidden              "$HIDDEN" \
        --n-layers            "$LAYERS" \
        --adv-capacity        "$ADV_CAP" \
        --policy-capacity     "$POL_CAP" \
        --ckpt-dir            "$CKPT_DIR" \
        --seed                "$SEED" \
        --checkpoint-every-iter 10
