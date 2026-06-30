#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Full retrain (align + task + θ_pred / Eq.8) under the corrected
# Analytic-Gaussian LDP, then blind evaluation. Seeded for reproducibility.
# ─────────────────────────────────────────────────────────────────────
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source ~/anaconda3/etc/profile.d/conda.sh && conda activate py358
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DATA="$PROJECT_ROOT/gui360_full/processed_data/action_prediction_train_resize/training_data.json"
IMG="$PROJECT_ROOT/gui360_full/processed_data/action_prediction_train_resize"

# Grid resolution G (M=G*G regions). Override with: GRID=16 bash train_eval_eq8.sh
GRID="${GRID:-10}"
CKPT="$PROJECT_ROOT/checkpoints/wproj_eq8_g${GRID}.pt"
LOG_DIR="$PROJECT_ROOT/results/eq8_g${GRID}"
mkdir -p "$LOG_DIR"

SEED=42
TRAIN_SAMPLES=5000
EVAL_SAMPLES=100
EVAL_EPS=5.0
EVAL_K=3
EVAL_TEMP=0.7

echo "[$(date '+%F %T')] ===== Stage 1-3 training (align2+task3+ltm3), grid=${GRID}x${GRID} ====="
stdbuf -oL -eL python -u -m hmdp.blind_vlm \
  --model qwen2.5-vl-7b --device cuda \
  --data-path "$DATA" --image-base "$IMG" \
  --num-samples "$TRAIN_SAMPLES" --grid "$GRID" \
  --align-epochs 2 --task-epochs 3 --ltm-epochs 3 \
  --lr 1e-4 --ltm-lr 1e-3 --c-star-sigma 0.2 \
  --seed "$SEED" --out "$CKPT" \
  > "$LOG_DIR/train.log" 2>&1
RC=$?
echo "TRAIN_DONE_$RC"
if [ "$RC" != "0" ] || [ ! -f "$CKPT" ]; then
  echo "[abort] training failed (rc=$RC)"; exit 1
fi

echo "[$(date '+%F %T')] ===== Blind evaluation (grid=$GRID eps=$EVAL_EPS k=$EVAL_K) ====="
stdbuf -oL -eL python -u -m hmdp.run_real_vlm \
  --device cuda --models qwen2.5-vl-7b \
  --num-samples "$EVAL_SAMPLES" --grid "$GRID" \
  --epsilon "$EVAL_EPS" --k "$EVAL_K" --temperature "$EVAL_TEMP" \
  --seed "$SEED" --wproj-ckpt "$CKPT" \
  --data-path "$DATA" --image-base "$IMG" \
  > "$LOG_DIR/eval.log" 2>&1
echo "EVAL_DONE_$?"
echo "[$(date '+%F %T')] ===== ALL DONE (grid=$GRID) ====="
