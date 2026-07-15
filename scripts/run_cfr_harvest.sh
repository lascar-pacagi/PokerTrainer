#!/usr/bin/env bash
# Offline average-policy harvest for a finished/paused cfr_coro run.
#
# Trains the deployable PolicyNet (average strategy) across all GPUs via DDP,
# reading the strategy reservoir shards (cfr_buffers_rank*.npz) and the model
# architecture from the run's checkpoint. Decoupled from training so it can run
# to convergence (big STEPS/BATCH) — see trainer/cfr/harvest_policy.py.
#
#   bash scripts/run_cfr_harvest.sh cfr.sif /scratch/$USER/cfr_run
#
# Defaults: NPROC=all GPUs, STEPS=200000, BATCH=8192, AMP=1, LR=run default.
# Override via env, e.g.:
#   STEPS=400000 BATCH=16384 WARM_START=1 bash scripts/run_cfr_harvest.sh cfr.sif /path
#
# ⚠ NPROC MUST MATCH the training run's GPU count. Each rank reads its own
# cfr_buffers_rank<rank>.npz; launching with fewer ranks silently drops shards,
# with more ranks fails on the missing shard file. Set NPROC explicitly if the
# harvest node has a different GPU count than the training node.
#
# NOTE: the image embeds /opt/pokertrainer/trainer at BUILD time — REBUILD the
# .sif (or bind-mount the repo) after changing trainer code, or it runs the old.
set -euo pipefail

SIF="${1:?usage: run_cfr_harvest.sh <path-to-sif> <ckpt-dir>}"
CKPT_DIR="${2:?ckpt-dir required (the finished/paused cfr_coro run directory)}"

STEPS="${STEPS:-200000}"
BATCH="${BATCH:-8192}"
AMP="${AMP:-1}"
LR="${LR:-}"                       # empty → harvest_policy uses the run's cfg.optim.lr
WARM_START="${WARM_START:-0}"      # 1 → init PolicyNet from the ckpt instead of fresh
OUT="${OUT:-cfr_policy_harvested.ckpt}"
SEED="${SEED:-12648430}"
DEVICE="${DEVICE:-cuda:0}"
MASTER_PORT="${MASTER_PORT:-29511}"

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[[ "$NPROC" -ge 1 ]] 2>/dev/null || NPROC=1

CKPT_DIR="$(readlink -f "$CKPT_DIR")"
export SINGULARITYENV_PT_ORACLE_CACHE="${PT_ORACLE_CACHE:-$CKPT_DIR/oracle_cache}"

FLAGS=(--ckpt-dir "$CKPT_DIR" --steps "$STEPS" --batch "$BATCH"
       --out "$OUT" --seed "$SEED" --device "$DEVICE")
[[ "$AMP" == "1" ]] && FLAGS+=(--amp)
[[ "$WARM_START" == "1" ]] && FLAGS+=(--warm-start)
[[ -n "$LR" ]] && FLAGS+=(--lr "$LR")

echo "[run_cfr_harvest] sif=$SIF ckpt=$CKPT_DIR"
echo "[run_cfr_harvest] steps=$STEPS batch=$BATCH amp=$AMP warm_start=$WARM_START out=$OUT"
echo "[run_cfr_harvest] ddp: nproc=$NPROC (MUST equal the training run's GPU count)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [[ "$NPROC" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC"
            --master_port="$MASTER_PORT" -m cfr.harvest_policy)
else
    LAUNCH=(python3 -m cfr.harvest_policy)
fi

singularity exec --nv \
    --bind "$CKPT_DIR:$CKPT_DIR" \
    "$SIF" \
    "${LAUNCH[@]}" "${FLAGS[@]}"
