#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Epsilon (noise) sweep — NO training. Runs the blind H-MDP eval on a
# fixed W_proj checkpoint across increasing epsilon (decreasing Gaussian
# noise sigma). If Avg Dist drops as epsilon grows, the LDP noise is the
# grounding bottleneck; if it stays flat even at the near-clean epsilon,
# the bottleneck is the proxy-encoder / W_proj reconstruction itself.
#
#   sigma(eps): 5->1.78  10->1.00  20->0.58  100->0.19  100000->~0.004
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
CKPT="${CKPT:-$PROJECT_ROOT/checkpoints/wproj_eq8.pt}"   # M=25 by default
GRID="${GRID:-5}"
SEED=42
SAMPLES="${SAMPLES:-80}"
K=3
TEMP=0.7

OUT_DIR="$PROJECT_ROOT/results/eps_sweep"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
SUMMARY="$OUT_DIR/summary.csv"
echo "epsilon,sigma,base_point,base_dist,hmdp_point,hmdp_dist,ltm_used" > "$SUMMARY"

EPSILONS=(5 10 20 50 100 1000 100000)

for eps in "${EPSILONS[@]}"; do
  log="$LOG_DIR/eps_${eps}.log"
  echo "[$(date '+%H:%M:%S')] eval epsilon=$eps (grid=$GRID)"
  stdbuf -oL -eL python -u -m hmdp.run_real_vlm \
    --device cuda --models qwen2.5-vl-7b \
    --num-samples "$SAMPLES" --grid "$GRID" \
    --epsilon "$eps" --k "$K" --temperature "$TEMP" \
    --seed "$SEED" --wproj-ckpt "$CKPT" \
    --data-path "$DATA" --image-base "$IMG" \
    > "$log" 2>&1

  sigma=$(python3 -c "from hmdp.ldp import analytic_gaussian_sigma as s; print(f'{s($eps,1e-5,2.0):.4f}')" 2>/dev/null || echo "NA")
  base_pt=$(grep -A4 "Base Results" "$log" | grep -oP 'Point Acc:\s*\K[0-9.]+' | tail -1)
  base_d=$(grep -A4 "Base Results" "$log" | grep -oP 'Avg Dist:\s*\K[0-9.]+' | tail -1)
  hmdp_pt=$(grep -A5 "H-MDP Results" "$log" | grep -oP 'Point Acc:\s*\K[0-9.]+' | tail -1)
  hmdp_d=$(grep -A5 "H-MDP Results" "$log" | grep -oP 'Avg Dist:\s*\K[0-9.]+' | tail -1)
  ltm=$(grep -oP 'ltm_used=\K[0-9]+' "$log" | tail -1)
  echo "${eps},${sigma},${base_pt:-NA},${base_d:-NA},${hmdp_pt:-NA},${hmdp_d:-NA},${ltm:-NA}" >> "$SUMMARY"
  echo "  eps=$eps sigma=$sigma  hmdp_point=${hmdp_pt:-NA}  hmdp_dist=${hmdp_d:-NA}"
done

echo "[$(date '+%H:%M:%S')] sweep done. Summary:"
cat "$SUMMARY"
