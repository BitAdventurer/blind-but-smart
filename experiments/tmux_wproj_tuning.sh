#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Tmux-based W_proj training + evaluation tuning pipeline
#
# Runs multiple W_proj training configurations sequentially and evaluates
# each checkpoint against the best (epsilon, k) found by the grid search.
# ─────────────────────────────────────────────────────────────────────

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source ~/anaconda3/etc/profile.d/conda.sh && conda activate py358
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

DATA="$PROJECT_ROOT/gui360_full/processed_data/action_prediction_train_resize/training_data.json"
IMG="$PROJECT_ROOT/gui360_full/processed_data/action_prediction_train_resize"

OUT_DIR="$PROJECT_ROOT/results/wproj_tuning"
LOG_DIR="$OUT_DIR/logs"
CKPT_DIR="$OUT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

# Default evaluation config (will be overridden by best grid-search config)
EVAL_EPS=5.0
EVAL_K=3
EVAL_TEMP=0.5
EVAL_SAMPLES=100

# Read best grid-search config if available
GRID_SUMMARY="$PROJECT_ROOT/results/epsilon_k_grid/summary.json"
if [ -f "$GRID_SUMMARY" ]; then
    echo "Reading best grid-search config from $GRID_SUMMARY"
    BEST=$(python3 - <<PYEOF
import json
with open("$GRID_SUMMARY") as f:
    data = json.load(f)
runs = [r for r in data.get("runs", []) if r.get("point_acc") is not None]
if not runs:
    print("")
else:
    best = max(runs, key=lambda r: r["point_acc"])
    print(f"{best['epsilon']} {best['k']} {best['temperature']}")
PYEOF
    )
    if [ -n "$BEST" ]; then
        read EVAL_EPS EVAL_K EVAL_TEMP <<< "$BEST"
        echo "Using best grid config: eps=$EVAL_EPS k=$EVAL_K temp=$EVAL_TEMP"
    fi
fi

# Training configurations to try
# Each: "name:num_samples:align_epochs:task_epochs:lr"
CONFIGS=(
    "baseline_v2:5000:2:3:1e-4"
    "more_epochs:5000:3:5:1e-4"
    "higher_lr:5000:2:3:5e-4"
    "more_data:10000:2:3:1e-4"
)

SUMMARY_FILE="$OUT_DIR/summary.json"
echo "{
  \"start_time\": \"$(date -Iseconds)\",
  \"eval_config\": {\"epsilon\": $EVAL_EPS, \"k\": $EVAL_K, \"temperature\": $EVAL_TEMP, \"num_samples\": $EVAL_SAMPLES},
  \"runs\": []
}" > "$SUMMARY_FILE"

run_id=0
for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r name num_samples align_epochs task_epochs lr <<< "$cfg"
    run_id=$((run_id + 1))
    train_log="$LOG_DIR/${name}_train.log"
    eval_log="$LOG_DIR/${name}_eval.log"
    ckpt="$CKPT_DIR/wproj_${name}.pt"

    echo "[$(date '+%H:%M:%S')] Run $run_id: $name"
    echo "  train: samples=$num_samples align=$align_epochs task=$task_epochs lr=$lr"
    echo "  eval: eps=$EVAL_EPS k=$EVAL_K temp=$EVAL_TEMP"

    # Training
    stdbuf -oL -eL python -u -m hmdp.blind_vlm \
        --model qwen2.5-vl-7b \
        --device cuda \
        --data-path "$DATA" \
        --image-base "$IMG" \
        --num-samples "$num_samples" \
        --align-epochs "$align_epochs" \
        --task-epochs "$task_epochs" \
        --lr "$lr" \
        --out "$ckpt" \
        > "$train_log" 2>&1
    train_rc=$?
    echo "TRAIN_DONE_$train_rc" >> "$train_log"

    if [ "$train_rc" != "0" ] || [ ! -f "$ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] Training failed for $name; skipping eval"
        continue
    fi

    # Evaluation
    stdbuf -oL -eL python -u -m hmdp.run_real_vlm \
        --device cuda \
        --models qwen2.5-vl-7b \
        --num-samples "$EVAL_SAMPLES" \
        --epsilon "$EVAL_EPS" \
        --k "$EVAL_K" \
        --temperature "$EVAL_TEMP" \
        --wproj-ckpt "$ckpt" \
        --data-path "$DATA" \
        --image-base "$IMG" \
        > "$eval_log" 2>&1
    eval_rc=$?
    echo "EVAL_DONE_$eval_rc" >> "$eval_log"

    # Extract metrics
    action_acc=$(grep -oP 'Action Acc:\s*\K[0-9.]+' "$eval_log" | tail -1 || echo "null")
    point_acc=$(grep -oP 'Point Acc:\s*\K[0-9.]+' "$eval_log" | tail -1 || echo "null")
    avg_dist=$(grep -oP 'Avg Dist:\s*\K[0-9.]+' "$eval_log" | tail -1 || echo "null")
    ltm_used=$(grep -oP 'LTM Episodes:\s*\K[0-9]+' "$eval_log" | tail -1 || echo "null")
    total_time=$(grep -oP 'Total time:\s*\K[0-9]+' "$eval_log" | tail -1 || echo "null")

    python3 - <<PYEOF
import json
with open("$SUMMARY_FILE", "r") as f:
    data = json.load(f)
data["runs"].append({
    "run_id": $run_id,
    "name": "$name",
    "num_samples": $num_samples,
    "align_epochs": $align_epochs,
    "task_epochs": $task_epochs,
    "lr": "$lr",
    "ckpt": "$ckpt",
    "eval_epsilon": $EVAL_EPS,
    "eval_k": $EVAL_K,
    "eval_temperature": $EVAL_TEMP,
    "eval_num_samples": $EVAL_SAMPLES,
    "action_acc": $action_acc if "$action_acc" != "null" else None,
    "point_acc": $point_acc if "$point_acc" != "null" else None,
    "avg_dist": $avg_dist if "$avg_dist" != "null" else None,
    "ltm_used": int($ltm_used) if "$ltm_used" != "null" else None,
    "total_time_s": int($total_time) if "$total_time" != "null" else None,
    "train_log": "$train_log",
    "eval_log": "$eval_log",
    "timestamp": "$(date -Iseconds)"
})
with open("$SUMMARY_FILE", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

    echo "[$(date '+%H:%M:%S')] Run $run_id done: act=$action_acc pt=$point_acc dist=$avg_dist"
    echo ""
done

echo "[$(date '+%H:%M:%S')] All W_proj tuning runs complete."
echo "Summary: $SUMMARY_FILE"
