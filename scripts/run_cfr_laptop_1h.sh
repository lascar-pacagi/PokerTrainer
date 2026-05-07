#!/usr/bin/env bash
# Laptop ~1-hour Deep CFR run on GPU learner + 8 CPU actors.
#
# Hardware target: 12-core laptop CPU + ~7 GB Pascal-class GPU. Net is sized
# small (hidden=128, layers=2 — the same defaults the laptop smoke used) so
# per-state forward is cheap; cluster runs use the bigger 512/8 net via
# scripts/run_cfr_cluster.sh.
#
# Time budget breakdown (rough):
#   * actor traversals (T=60 × K=300 = 18,000 traversals)  ≈ 3000s @ ~6 traversals/s aggregate
#   * AdvNet refits (60 iters × 2 nets × 1000 grad steps)  ≈ 500s on cuda:0
#   * final PolicyNet training (20,000 grad steps)         ≈ 100s on cuda:0
#   ──────────────────────────────────────────────────────────────────
#   total                                                  ≈ 3600s (~1 hour)
#
# Override any of these via env vars: ACTORS=10 ITERS=40 K=500 ./run_cfr_laptop_1h.sh
#
# Output goes to runs/cfr_laptop_1h/ by default. The trainer writes
# checkpoints every 10 iterations + a `cfr_final.ckpt` at the end. Eval after
# the run with:
#     PYTHONPATH=engine/build:trainer python -m evaluate.cfr_ckpt \
#         --ckpt runs/cfr_laptop_1h/cfr_final.ckpt --hands 2000 \
#         --opponents random,check_fold,calling_station

set -euo pipefail

cd "$(dirname "$0")/.."

ACTORS="${ACTORS:-8}"
ITERS="${ITERS:-60}"
K="${K:-300}"
ADV_GRAD_STEPS="${ADV_GRAD_STEPS:-1000}"
POLICY_GRAD_STEPS="${POLICY_GRAD_STEPS:-20000}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-2}"
ADV_CAP="${ADV_CAP:-500000}"
POL_CAP="${POL_CAP:-1000000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
MAX_DEPTH="${MAX_DEPTH:-34}"
CKPT_DIR="${CKPT_DIR:-runs/cfr_laptop_1h}"

mkdir -p "$CKPT_DIR"
LOG="$CKPT_DIR/run.log"

echo "[run_cfr_laptop_1h] target wall ≈ 1 hour"
echo "[run_cfr_laptop_1h] actors=$ACTORS iters=$ITERS K=$K  device=$DEVICE"
echo "[run_cfr_laptop_1h] net hidden=$HIDDEN layers=$LAYERS  max_depth=$MAX_DEPTH"
echo "[run_cfr_laptop_1h] ckpt_dir=$CKPT_DIR"
echo "[run_cfr_laptop_1h] log file: $LOG"

PYTHONPATH=engine/build:trainer \
python -u -m cfr.cfr_mp_gpu \
    --mp-actors             "$ACTORS" \
    --learner-device        "$DEVICE" \
    --n-iterations          "$ITERS" \
    --n-traversals-per-iter "$K" \
    --adv-grad-steps        "$ADV_GRAD_STEPS" \
    --policy-grad-steps     "$POLICY_GRAD_STEPS" \
    --hidden                "$HIDDEN" \
    --n-layers              "$LAYERS" \
    --adv-capacity          "$ADV_CAP" \
    --policy-capacity       "$POL_CAP" \
    --ckpt-dir              "$CKPT_DIR" \
    --seed                  "$SEED" \
    --max-depth             "$MAX_DEPTH" \
    --checkpoint-every-iter 20 \
    --watchdog-every-s      30.0 \
    2>&1 | tee "$LOG"

echo "[run_cfr_laptop_1h] DONE. final ckpt: $CKPT_DIR/cfr_final.ckpt"
echo "[run_cfr_laptop_1h] eval suggestion:"
echo "    PYTHONPATH=engine/build:trainer python -m evaluate.cfr_ckpt \\"
echo "        --ckpt $CKPT_DIR/cfr_final.ckpt --hands 2000 \\"
echo "        --opponents random,check_fold,calling_station"
