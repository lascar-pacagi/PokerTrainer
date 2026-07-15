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
# Recommended defaults are baked in (multi-GPU DDP + AMP + atomic resume), so the
# bare command is the recommended run:
#   bash scripts/run_cfr_cluster.sh cfr.sif /scratch/$USER/cfr_run
# Defaults: NPROC=all GPUs, AMP=1, ADV_GRAD_STEPS=3000 (DDP) / 5000 (1 GPU),
#           MAX_RAISES=3, SAVE_BUFFERS=1, CKPT_EVERY=20.
# Override any via env, e.g.:
#   AMP=0 ADV_GRAD_STEPS=5000 CKPT_EVERY=50 bash scripts/run_cfr_cluster.sh ...
#
# This trains the AdvNets only. The deployable average policy is harvested
# separately from the saved buffers (keep SAVE_BUFFERS=1):
#   bash scripts/run_cfr_harvest.sh cfr.sif /scratch/$USER/cfr_run
#
# CFR_PG_TIMEOUT_S (default 30min) cushions the end-of-iteration barrier while
# rank 0 does its solo probe + checkpoint save (seconds normally). Raise it only
# if checkpointing a very large run is slow.
#
# Resume is AUTOMATIC: re-launching with the SAME ckpt-dir (and SAME NPROC, see
# below) picks up the latest cfr_iter_*.ckpt + per-rank cfr_buffers_rank*.npz and
# continues. A fresh run just needs an empty ckpt-dir.
#
# ⚠ Keep NPROC CONSTANT across restarts for an exact resume. The reservoir
# buffers are sharded one-file-per-rank, so changing the GPU count orphans the
# shards whose rank index no longer exists (3→2 GPU drops cfr_buffers_rank2.npz,
# ~1/N of the data) or leaves new ranks with empty buffers (2→3 GPU). The NETS
# always resume regardless; only buffer data is lost, and the refit warm-up
# protects the loaded nets while the kept shards refill. If you expect to vary
# the GPU count, run SAVE_BUFFERS=0 (nets-only resume, no shard mismatch).
#
# NOTE: the image embeds /opt/pokertrainer/trainer at BUILD time. After changing
# trainer code, REBUILD the .sif (or bind-mount the repo over the container path)
# or the container will run the old code.
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
# ADV_GRAD_STEPS default is decided AFTER NPROC is known (below): 3000 under DDP
# — where each step sees NPROC×batch, so 3000 steps still refit on more data than
# the old single-GPU 5000 — vs 5000 single-GPU (3000 there would under-fit).
ADV_GRAD_STEPS="${ADV_GRAD_STEPS:-}"
# weight_decay default 0: a controlled diagnostic showed 1e-3 freezes the AdvNet
# to a constant action (kills hole-card discrimination). See cfr_coro.py.
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
D_MODEL="${D_MODEL:-512}"
LAYERS="${LAYERS:-8}"
N_HEADS="${N_HEADS:-12}"
D_FF="${D_FF:-2048}"
ADV_CAP="${ADV_CAP:-20000000}"
POL_CAP="${POL_CAP:-40000000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

# ── Multi-GPU (DDP) ─────────────────────────────────────────────────────────
# NPROC>1 launches one process per GPU via torchrun; each rank pins cuda:LOCAL_RANK,
# shards collection, and the AdvNet refit (the ~95% bottleneck) runs data-parallel
# with all-reduced gradients → ~NPROC× refit throughput + effective batch
# NPROC×batch. NPROC=1 keeps the exact single-process path (plain python -m).
# Default: all visible GPUs. NOTE on heterogeneous GPUs (e.g. A40 + RTX 8000):
# DDP runs at the SLOWEST GPU's pace; if NCCL stalls across arch generations, try
# NCCL_P2P_DISABLE=1. To use only the matched GPUs, set CUDA_VISIBLE_DEVICES.
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" -ge 1 ]] 2>/dev/null || NPROC=1
MASTER_PORT="${MASTER_PORT:-29500}"

# Resolve the refit-step default now that NPROC is known (see note above).
if [[ -z "$ADV_GRAD_STEPS" ]]; then
    if [[ "$NPROC" -gt 1 ]]; then ADV_GRAD_STEPS=3000; else ADV_GRAD_STEPS=5000; fi
fi

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

# ── Tree-size + buffer knobs ────────────────────────────────────────────────
# MAX_RAISES caps voluntary raises per street (training-only action abstraction)
# — bounds the deep-stack re-raise wars that make full-depth 100bb traversal
# explode. 0 = uncapped. SAVE_BUFFERS=1 persists the reservoirs with each
# checkpoint for an exact, zero-regression resume AND so the average policy can
# be harvested offline later (scripts/run_cfr_harvest.sh) — keep it on. Multi-GB;
# raise CKPT_EVERY if I/O dominates.
MAX_RAISES="${MAX_RAISES:-3}"
SAVE_BUFFERS="${SAVE_BUFFERS:-1}"
# CKPT_EVERY=20: a cluster wall-time kill loses up to this many iterations, so
# keep it modest for the 15-day-limit → restart workflow (the buffer write is a
# few seconds on scratch). Raise it only if buffer I/O is a measured problem.
CKPT_EVERY="${CKPT_EVERY:-20}"
# AMP=1 → fp16 mixed precision on the refit/policy training (~1.5-2× on top of
# DDP; fp16 is portable across the Turing+Ampere mix). On by default; AMP=0 off.
AMP="${AMP:-1}"

EXTRA_FLAGS=(--max-raises-per-street "$MAX_RAISES"
             --checkpoint-every-iter "$CKPT_EVERY")
[[ "$SAVE_BUFFERS" == "1" ]] || EXTRA_FLAGS+=(--no-save-buffers)
[[ "$AMP" == "1" ]] && EXTRA_FLAGS+=(--amp)
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
echo "[run_cfr_cluster] tree: max_raises=$MAX_RAISES save_buffers=$SAVE_BUFFERS ckpt_every=$CKPT_EVERY"
echo "[run_cfr_cluster] ddp: nproc=$NPROC (1 ⇒ single-process; >1 ⇒ torchrun multi-GPU)"
echo "[run_cfr_cluster] resume: auto from latest cfr_iter_*.ckpt in ckpt-dir (empty dir ⇒ fresh)"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv || true

# NPROC>1 → torchrun (one rank per GPU); NPROC==1 → plain python (unchanged path).
if [[ "$NPROC" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
            --master_port="$MASTER_PORT" -m cfr.cfr_coro)
else
    LAUNCH=(python3 -m cfr.cfr_coro)
fi

singularity exec --nv \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    "${LAUNCH[@]}" \
        --n-virtual           "$ACTORS" \
        --device              "$DEVICE" \
        --n-iterations        "$ITERS" \
        --n-traversals-per-iter "$K" \
        --adv-grad-steps      "$ADV_GRAD_STEPS" \
        --weight-decay        "$WEIGHT_DECAY" \
        --d-model             "$D_MODEL" \
        --n-layers            "$LAYERS" \
        --n-heads             "$N_HEADS" \
        --d-ff                "$D_FF" \
        --adv-capacity        "$ADV_CAP" \
        --policy-capacity     "$POL_CAP" \
        --ckpt-dir            "$CKPT_DIR" \
        --seed                "$SEED" \
        "${EXTRA_FLAGS[@]}" \
        "${STAGE_FLAGS[@]}"
