#!/bin/bash
set -u

# Train a fresh Eq.8 W_proj checkpoint and immediately run a small blind
# evaluation. This script is intentionally conservative for GitHub use:
# it writes a new checkpoint by default and never overwrites the checked-in
# documentation or canonical paper-result path.

# Resolve this script's directory and use it as the project root so the script
# works from any current working directory.
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_ROOT"
source ~/anaconda3/etc/profile.d/conda.sh && conda activate py358
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="$CODE_ROOT/gui360_full/processed_data/action_prediction_train_resize/training_data.json"
IMG="$CODE_ROOT/gui360_full/processed_data/action_prediction_train_resize"
CKPT="${CKPT:-$CODE_ROOT/checkpoints/wproj_eq8_run.pt}"
LOG_DIR="$CODE_ROOT/logs"
RESULT_JSON="$CODE_ROOT/results/json/run_train_eval_results.json"
mkdir -p "$LOG_DIR"

# Add code directory to PYTHONPATH for module imports.
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"

echo "=== [1/2] Offline W_proj training (Eq. 8 LTM predictor enabled) ===" | tee "$LOG_DIR/train_eq8_run.log"
stdbuf -oL -eL python -u -m hmdp.blind_vlm --model qwen2.5-vl-7b --device cuda \
  --data-path "$DATA" --image-base "$IMG" \
  --num-samples 5000 --align-epochs 2 --task-epochs 3 --lr 1e-4 \
  --out "$CKPT" >> "$LOG_DIR/train_eq8_run.log" 2>&1
RC=$?
echo "TRAIN_DONE_$RC" >> "$LOG_DIR/train_eq8_run.log"
if [ "$RC" != "0" ] || [ ! -f "$CKPT" ]; then echo "[abort] training failed"; exit 1; fi

# Fail fast before the expensive VLM launch if data/checkpoint paths are wrong.
python validate_runtime.py --data-path "$DATA" --image-base "$IMG" --wproj-ckpt "$CKPT"

echo "=== [2/2] Blind evaluation ===" | tee "$LOG_DIR/eval_eq8_run.log"
stdbuf -oL -eL python -u -m hmdp.run_real_vlm --wproj-ckpt "$CKPT" \
  --models qwen2.5-vl-7b --num-samples 100 --epsilon 1.0 --k 5 \
  --data-path "$DATA" --image-base "$IMG" \
  --out "$RESULT_JSON" \
  >> "$LOG_DIR/eval_eq8_run.log" 2>&1
echo "EVAL_DONE_$?" >> "$LOG_DIR/eval_eq8_run.log"
