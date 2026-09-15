# Blind but Smart: Hierarchical Joint Control of Perception Budget and Reasoning Depth in Agentic GUI Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### Overview

This repository contains the reference implementation and experiment harness for
**Blind but Smart**, an H-MDP based framework for agentic GUI systems. The core
idea is to jointly control:

- the **perception privacy budget** used for local differential privacy (LDP),
- the **reasoning depth** used by graph-of-thought style VLM inference,
- and the **long-term memory** refinement path used under uncertain grounding.

Large local artifacts are intentionally excluded from git: GUI-360 data,
screenshots, checkpoints, generated results, logs, figures, reports, and local
paper PDFs.

**Key Features**

- **Blind-but-Smart VLM Boundary**: the main path injects privatized DINOv2
  latents through a learned W_proj layer instead of sending raw screenshots.
- **Local Differential Privacy**: per-region Gaussian LDP over a GxG screen
  partition.
- **Graph-of-Thought Reasoning**: k-path VLM sampling and aggregation for GUI
  action/coordinate prediction.
- **Long-Term Memory Refinement**: Eq.8-style coordinate refinement from stored
  interaction priors when uncertainty is high.
- **Adaptive Epsilon Governor**: optional SAC meta-policy support for per-region
  epsilon and reasoning-depth control.
- **Reproducible Experiment Harness**: real VLM runners, missing-table scripts,
  appendix runners, report generators, and runtime preflight checks.

## Project Structure

```text
blind-but-smart/
├── hmdp/                         # Main real-VLM H-MDP implementation
│   ├── blind_vlm.py              # Blind-but-Smart latent-injection pipeline
│   ├── run_real_vlm.py           # Real VLM base vs H-MDP evaluation
│   ├── run_all_tables.py         # Main table harness
│   ├── run_missing_tables.py     # Additional paper-table experiments
│   ├── sac_governor.py           # Adaptive epsilon/k governor
│   ├── ldp.py                    # Local differential privacy modules
│   ├── ltm.py                    # Long-term memory module
│   ├── dinov2_encoder.py         # DINOv2 full-image and region encoders
│   ├── got/                      # Graph-of-thought path generation/scoring
│   ├── projection/               # W_proj, task head, Eq.8 LTM predictor
│   └── vlm_adapters/             # Qwen/InternVL loading helpers
│
├── hmdp_sim/                     # Simulation and SAC-style training components
├── experiments/                  # Grid search, epsilon sweep, W_proj tuning
├── docs/                         # Usage notes and implementation notes
├── checkpoints/                  # Local checkpoints placeholder (ignored)
├── gui360_full/                  # Local GUI-360 data placeholder (ignored)
├── gui360_bench/                 # Local GUI-360-Bench placeholder (ignored)
├── results/                      # Generated results placeholder (ignored)
├── logs/                         # Local logs placeholder (ignored)
├── validate_runtime.py           # Fast preflight validator
├── run_train_eval.sh             # W_proj training + small blind eval script
├── test_adaptive_epsilon.py      # Governor smoke tests
├── requirements.txt              # Public Python dependencies
└── README.md                     # Project documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BitAdventurer/blind-but-smart.git
cd blind-but-smart

# Create/activate your Python environment, then install dependencies
pip install -r requirements.txt
```

If using the original server environment:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py358
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Data and Checkpoints

Place GUI-360 action prediction data under:

```text
gui360_full/processed_data/action_prediction_train_resize/
```

The default training JSON path is:

```text
gui360_full/processed_data/action_prediction_train_resize/training_data.json
```

The main Blind-but-Smart checkpoint expected by default is:

```text
checkpoints/wproj_eq8.pt
```

This checkpoint should include:

- `proj_layer`
- `proxy_encoder`
- `task_head`
- optional but recommended `ltm_predictor`

### Runtime Preflight

Before launching a GPU/VLM run, validate local paths and checkpoint structure:

```bash
python validate_runtime.py
```

This avoids long VLM launches failing late because of missing data, stale split
paths, broken image roots, or corrupted checkpoints.

### Real VLM Smoke Test

```bash
python -m hmdp.run_real_vlm \
  --device cuda \
  --models qwen2.5-vl-7b \
  --num-samples 1 \
  --epsilon 1.0 \
  --k 1 \
  --wproj-ckpt checkpoints/wproj_eq8.pt \
  --out results/json/smoke_real_vlm.json
```

### Larger Real VLM Evaluation

```bash
python -m hmdp.run_real_vlm \
  --device cuda \
  --models qwen2.5-vl-7b \
  --num-samples 100 \
  --epsilon 1.0 \
  --k 5 \
  --wproj-ckpt checkpoints/wproj_eq8.pt \
  --out results/json/real_vlm_hmdp_results.json
```

## Detailed Usage

### 1. Blind-but-Smart Evaluation

The default path in `hmdp.run_real_vlm` is the paper-faithful Blind-but-Smart
latent-injection path:

```bash
python -m hmdp.run_real_vlm \
  --device cuda \
  --models qwen2.5-vl-7b \
  --epsilon 1.0 \
  --k 5 \
  --wproj-ckpt checkpoints/wproj_eq8.pt
```

Use `--raw-pixel` only for ablation. It gives the original screenshot to the VLM
and therefore breaks the Blind-but-Smart privacy boundary.

### 2. W_proj Training and Evaluation

Train a fresh Eq.8 W_proj checkpoint and run a small blind evaluation:

```bash
bash run_train_eval.sh
```

By default this writes:

```text
checkpoints/wproj_eq8_run.pt
results/json/run_train_eval_results.json
```

Override the checkpoint output path:

```bash
CKPT=/path/to/wproj_custom.pt bash run_train_eval.sh
```

### 3. Main Paper Table Harness

Run selected numbered tables:

```bash
python -m hmdp.run_all_tables \
  --device cuda \
  --table 1 2 \
  --table1-model qwen2.5-vl-7b \
  --num-samples 150 \
  --wproj-ckpt checkpoints/wproj_eq8.pt \
  --out results/json/real_all_tables.json
```

Run appendix-style table IDs:

```bash
python -m hmdp.run_all_tables \
  --device cuda \
  --table-str b9 b12 \
  --wproj-ckpt checkpoints/wproj_eq8.pt \
  --out results/json/real_appendix_tables.json
```

### 4. Adaptive Epsilon Governor

Enable the SAC meta-policy governor:

```bash
python -m hmdp.run_real_vlm \
  --device cuda \
  --models qwen2.5-vl-7b \
  --adaptive \
  --sac-ckpt /path/to/sac_meta_policy.pt \
  --wproj-ckpt checkpoints/wproj_eq8.pt
```

If the SAC checkpoint is missing or fails `torch.load`, the governor degrades to
the configured uniform epsilon fallback with a warning.

### 5. Report Generation

Generate paper-style text/LaTeX table output from a result JSON:

```bash
python generate_paper_results.py \
  --results results/json/real_vlm_hmdp_results.json \
  --out results/json/paper_table_3.txt
```

Generate a comprehensive markdown/text report:

```bash
python generate_comprehensive_results.py \
  --results results/json/real_vlm_hmdp_results.json \
  --report-out results/json/evaluation_report.md \
  --out results/json/comprehensive_results.txt
```

## Core Components

### Blind VLM Pipeline

`hmdp/blind_vlm.py` implements the execution boundary where raw pixels never
reach the VLM. Screens are partitioned into regions, encoded with DINOv2,
privatized with LDP noise, projected through W_proj, and injected as visual
prefix embeddings.

### LDP Mechanism

`hmdp/ldp.py` contains the proxy encoder and Gaussian local differential privacy
mechanism used for per-region latent perturbation.

### GoT Reasoning

`hmdp/got/` contains path generation, path scoring, and aggregation logic for
k-path reasoning over GUI action/coordinate predictions.

### Long-Term Memory

`hmdp/ltm.py` stores successful or near-successful episodes and retrieves
spatial priors for uncertainty-triggered coordinate refinement.

### SAC Governor

`hmdp/sac_governor.py` wraps the recovered SAC meta-policy and provides robust
fallback behavior when checkpoints are absent, corrupted, or structurally
incompatible.

## Data Policy

The repository intentionally does not include:

- GUI-360 data or screenshots
- GUI-360-Bench data
- trained `.pt` / `.pth` / `.ckpt` checkpoints
- generated result JSON files
- figures, logs, reports, or local PDFs

These paths are represented by `.gitkeep` placeholders and ignored by
`.gitignore`. Keep large artifacts outside git or distribute them separately.

## Validation

Recommended checks before pushing changes:

```bash
python -m compileall -q .
python test_adaptive_epsilon.py
python validate_runtime.py
git status --short --ignored
```

Confirm that only source code, documentation, and placeholders are tracked.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
