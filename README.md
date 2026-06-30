# Blind but Smart

Hierarchical Joint Control of Perception Budget and Reasoning Depth in Agentic
GUI Systems.

This repository contains the reference implementation, experiment harness, and
lightweight documentation for the ESWA2 H-MDP GUI-agent experiments. Large
local artifacts are intentionally excluded from git:

- GUI-360 data and screenshots
- trained projection / SAC checkpoints
- generated result JSON, logs, figures, and reports
- local paper PDFs

## Environment

```bash
cd /path/to/ESWA2/code
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py358
export PYTHONPATH=$PWD:$PYTHONPATH
```

The server default PATH may not provide `python`; activate `py358` first.

For a clean clone, install the public Python dependencies:

```bash
pip install -r requirements.txt
```

## Main Data

GUI-360 training data and images live under:

```text
gui360_full/processed_data/action_prediction_train_resize/
```

The main training JSON is:

```text
gui360_full/processed_data/action_prediction_train_resize/training_data.json
```

Data files, generated results, logs, PDFs, and model checkpoints are ignored by
git. After cloning, place local GUI-360 data and checkpoints back under the
paths above, or point scripts at external locations with the documented CLI
flags / environment variables.

Expected checkpoint for the main Blind-but-Smart path:

```text
checkpoints/wproj_eq8.pt
```

## Main Commands

Run a fast local sanity check before launching a GPU job:

```bash
python validate_runtime.py
```

Run a real VLM / H-MDP smoke evaluation:

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

Run a larger real VLM / H-MDP evaluation:

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

Train the blind W_proj projection and run a small blind evaluation:

```bash
bash run_train_eval.sh
```

By default this writes a fresh checkpoint to `checkpoints/wproj_eq8_run.pt`.
Override with `CKPT=/path/to/out.pt bash run_train_eval.sh` when needed.

The default Blind-but-Smart path uses `checkpoints/wproj_eq8.pt` unless
`--wproj-ckpt` is overridden. For a quick smoke test without a trained
projection checkpoint, pass `--allow-random-wproj`; do not use that mode for
paper results.

Full evaluation defaults to `results/json/real_vlm_hmdp_results.json`.
Short smoke runs should keep using an explicit `--out results/json/smoke_*.json`
path so they do not overwrite paper-result files.

The local `checkpoints/wproj_eq8.pt` checkpoint includes the projection layer,
task head, and LTM predictor. Older checkpoints may omit the LTM predictor, and
corrupted checkpoint files should not be used for reported results.

Generate ESWA paper tables/figures from scripted result values:

```bash
python run_eswa_tables.py
```

## Directory Map

```text
hmdp/          Main implementation: Blind VLM, LDP, GoT, LTM, real VLM eval
hmdp_sim/      Simulation implementation and SAC-style policy components
gui360_full/   Local GUI-360 data placeholder; contents are git-ignored
gui360_bench/  Local GUI-360-Bench data placeholder; contents are git-ignored
checkpoints/   Local model/projection checkpoints; contents are git-ignored
experiments/   Experiment/tuning scripts
results/       Generated results placeholder; contents are git-ignored
logs/          Local run logs placeholder; contents are git-ignored
docs/          Notes and usage docs
```

## Notes

- `run_train_eval.sh` assumes this `code/` directory is the project root.
- The default `hmdp.run_real_vlm` path is Blind-but-Smart latent injection; use
  `--raw-pixel` only for ablation.
- Adaptive SAC mode (`--adaptive --sac-ckpt ...`) requires a checkpoint that
  passes `torch.load`; the bundled `checkpoints/hmdp/*.pt` files should be
  treated as invalid unless re-generated.
- Top-level analysis/generation scripts are kept at the root for now to avoid breaking relative imports and paths.

## Release Checklist

Before pushing to GitHub:

```bash
python -m compileall -q .
python test_adaptive_epsilon.py
python validate_runtime.py
git status --short --ignored
```

Confirm that only source/docs/placeholders are staged. Data, checkpoints,
results, logs, caches, and PDFs should appear as ignored files.
