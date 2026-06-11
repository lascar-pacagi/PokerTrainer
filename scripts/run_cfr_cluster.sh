#!/usr/bin/env bash
# Cluster launch wrapper for Deep CFR training (cfr.cfr_coro) inside the
# Singularity image. ACTORS env var maps to cfr_coro's --n-virtual flag
# (number of concurrent coroutines / GPU batch size for inference); the
# 16-64 sweet spot in cfr_coro.py's docstring applies on Turing.
#
# Model is a token-sequence transformer (see project_cfr_token_architecture).
# Default size below is small (~830k params) to validate the architecture's
# hole-card-conditioning on the cluster cheaply; scale up once verified.
#
# Tested on rtx-8000 (sm_75), A40 (sm_86), L40S (sm_89). All three work with
# the same .sif (cu124 wheel falls back to fp32/fp16 on Turing — no bf16, but
# the per-iter compute is small enough that it doesn't matter).
#
# Usage:
#   bash scripts/run_cfr_cluster.sh /path/to/pokertrainer-cfr.sif /scratch/$USER/cfr_run
#
# Override defaults via env:
#   ACTORS=32 ITERS=500 K=10000 D_MODEL=256 LAYERS=6 \
#       bash scripts/run_cfr_cluster.sh ...
#
# Pre-flight checks expected on the cluster:
#   * a CUDA-visible GPU (nvidia-smi works inside `singularity exec --nv`)
#   * /scratch/$USER writable
#   * --bind /scratch if the image was built without it accessible
set -euo pipefail

SIF="${1:?usage: run_cfr_cluster.sh <path-to-sif> <ckpt-dir>}"
CKPT_DIR="${2:?ckpt-dir required (e.g. /scratch/$USER/cfr_run)}"

# Defaults sized for the architecture-validation run on RTX 8000.
# Tiny ~830k-param transformer (d_model=128, layers=4, heads=4) — meant to
# answer "does the new architecture actually condition on hole cards?" in
# the cheapest possible cluster run. Per-iter wall expected <5 min at
# K=1000; 1000 iters in ~3 days. Scale up to D_MODEL=256 + LAYERS=6 once
# the iter-50 sensitivity diagnostic confirms σ-range across 38 hands > 0.20.
ACTORS="${ACTORS:-64}"
ITERS="${ITERS:-1000}"
K="${K:-1000}"
ADV_GRAD_STEPS="${ADV_GRAD_STEPS:-5000}"
POLICY_GRAD_STEPS="${POLICY_GRAD_STEPS:-50000}"
# weight_decay default 0: a controlled diagnostic showed 1e-3 freezes the AdvNet
# to a constant action (kills hole-card discrimination). See cfr_coro.py.
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
D_MODEL="${D_MODEL:-128}"
LAYERS="${LAYERS:-4}"
N_HEADS="${N_HEADS:-4}"
D_FF="${D_FF:-512}"
ADV_CAP="${ADV_CAP:-20000000}"
POL_CAP="${POL_CAP:-40000000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

# ── Curriculum stage 1: short-stack push/fold ───────────────────────────────
# STACK_BB sets the effective starting stack (bb). PUSH_FOLD=1 restricts the
# traversal to FOLD/CALL/ALL_IN and switches the per-iter diagnostic to the
# Nash push/fold range validation (net jam/call grids vs the solved oracle).
#   STACK_BB=10 PUSH_FOLD=1 bash scripts/run_cfr_cluster.sh <sif> <ckpt>
# The Nash oracle is solved once from an ORACLE_DEALS-deal equity matrix and
# cached under $CKPT_DIR/oracle_cache (PT_ORACLE_CACHE).
STACK_BB="${STACK_BB:-100}"
PUSH_FOLD="${PUSH_FOLD:-0}"
ORACLE_DEALS="${ORACLE_DEALS:-12000000}"
# SINGULARITYENV_ prefix is the reliable way to inject an env var into the
# container; it appears inside as PT_ORACLE_CACHE. Point it at the bound,
# writable ckpt dir (the container CWD's runs/ may be read-only).
export SINGULARITYENV_PT_ORACLE_CACHE="${PT_ORACLE_CACHE:-$CKPT_DIR/oracle_cache}"

STAGE_FLAGS=(--starting-stack-bb "$STACK_BB")
if [[ "$PUSH_FOLD" == "1" ]]; then
    STAGE_FLAGS+=(--push-fold --oracle-deals "$ORACLE_DEALS")
fi

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
echo "[run_cfr_cluster] d_model=$D_MODEL layers=$LAYERS heads=$N_HEADS d_ff=$D_FF device=$DEVICE"
echo "[run_cfr_cluster] stage: stack_bb=$STACK_BB push_fold=$PUSH_FOLD oracle_deals=$ORACLE_DEALS"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

singularity exec --nv \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    python3 -m cfr.cfr_coro \
        --n-virtual           "$ACTORS" \
        --device              "$DEVICE" \
        --n-iterations        "$ITERS" \
        --n-traversals-per-iter "$K" \
        --adv-grad-steps      "$ADV_GRAD_STEPS" \
        --policy-grad-steps   "$POLICY_GRAD_STEPS" \
        --weight-decay        "$WEIGHT_DECAY" \
        --d-model             "$D_MODEL" \
        --n-layers            "$LAYERS" \
        --n-heads             "$N_HEADS" \
        --d-ff                "$D_FF" \
        --adv-capacity        "$ADV_CAP" \
        --policy-capacity     "$POL_CAP" \
        --ckpt-dir            "$CKPT_DIR" \
        --seed                "$SEED" \
        --checkpoint-every-iter 10 \
        "${STAGE_FLAGS[@]}"
