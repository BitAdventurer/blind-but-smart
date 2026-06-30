#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Tmux-based Epsilon / k Grid Search for H-MDP Real-VLM Evaluation
#
# Goal: find the best (epsilon, k, temperature) configuration for the
# Blind-but-Smart H-MDP pipeline using existing W_proj checkpoints.
# Results are aggregated into a JSON file for easy analysis.
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

# Checkpoints to test (current Eq.8 default and Gaussian baseline)
WPROJ_CKPTS=(
  "$PROJECT_ROOT/checkpoints/wproj_eq8.pt"
  "$PROJECT_ROOT/checkpoints/wproj_gaussian.pt"
)

# Grid configurations
EPSILONS=(1.0 5.0 10.0)
KS=(1 3 5 10)
TEMPERATURES=(0.5 0.7)
NUM_SAMPLES=50
MODEL="qwen2.5-vl-7b"

OUT_DIR="$PROJECT_ROOT/results/epsilon_k_grid"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

SUMMARY_FILE="$OUT_DIR/summary.json"

echo "{
  \"start_time\": \"$(date -Iseconds)\",
  \"num_samples\": $NUM_SAMPLES,
  \"model\": \"$MODEL\",
  \"runs\": []
}" > "$SUMMARY_FILE"

run_id=0
for ckpt in "${WPROJ_CKPTS[@]}"; do
  ckpt_name=$(basename "$ckpt" .pt)
  for eps in "${EPSILONS[@]}"; do
    for k in "${KS[@]}"; do
      for temp in "${TEMPERATURES[@]}"; do
        run_id=$((run_id + 1))
        run_name="${ckpt_name}_eps${eps}_k${k}_t${temp}"
        log_file="$LOG_DIR/${run_name}.log"
        out_file="$OUT_DIR/${run_name}.json"

        echo "[$(date '+%H:%M:%S')] Run $run_id: $run_name"
        echo "  ckpt=$ckpt, eps=$eps, k=$k, temp=$temp"

        stdbuf -oL -eL python -u -m hmdp.run_real_vlm \
          --device cuda \
          --models "$MODEL" \
          --num-samples "$NUM_SAMPLES" \
          --epsilon "$eps" \
          --k "$k" \
          --temperature "$temp" \
          --wproj-ckpt "$ckpt" \
          --data-path "$DATA" \
          --image-base "$IMG" \
          > "$log_file" 2>&1

        # Extract key metrics from the log
        action_acc=$(grep -oP 'Action Acc:\s*\K[0-9.]+' "$log_file" | tail -1 || echo "null")
        point_acc=$(grep -oP 'Point Acc:\s*\K[0-9.]+' "$log_file" | tail -1 || echo "null")
        avg_dist=$(grep -oP 'Avg Dist:\s*\K[0-9.]+' "$log_file" | tail -1 || echo "null")
        ltm_used=$(grep -oP 'LTM Episodes:\s*\K[0-9]+' "$log_file" | tail -1 || echo "null")
        total_time=$(grep -oP 'Total time:\s*\K[0-9]+' "$log_file" | tail -1 || echo "null")

        # Append to summary JSON
        python3 - <<PYEOF
import json
with open("$SUMMARY_FILE", "r") as f:
    data = json.load(f)
data["runs"].append({
    "run_id": $run_id,
    "run_name": "$run_name",
    "ckpt": "$ckpt",
    "ckpt_name": "$ckpt_name",
    "epsilon": $eps,
    "k": $k,
    "temperature": $temp,
    "num_samples": $NUM_SAMPLES,
    "action_acc": $action_acc if "$action_acc" != "null" else None,
    "point_acc": $point_acc if "$point_acc" != "null" else None,
    "avg_dist": $avg_dist if "$avg_dist" != "null" else None,
    "ltm_used": int($ltm_used) if "$ltm_used" != "null" else None,
    "total_time_s": int($total_time) if "$total_time" != "null" else None,
    "log_file": "$log_file",
    "timestamp": "$(date -Iseconds)"
})
with open("$SUMMARY_FILE", "w") as f:
    json.dump(data, f, indent=2)
PYEOF

        echo "[$(date '+%H:%M:%S')] Run $run_id done: act=$action_acc pt=$point_acc dist=$avg_dist"
        echo ""
      done
    done
  done
done

echo "[$(date '+%H:%M:%S')] All grid-search runs complete."
echo "Summary: $SUMMARY_FILE"
