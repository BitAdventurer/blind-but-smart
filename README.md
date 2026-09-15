# Blind but Smart

Reference implementation of joint regional disclosure-budget and candidate-count
control for recorded GUI grounding. The maintained runtime is
[`src/gui_joint_control`](src/gui_joint_control), exposed through the `bbs` command.

This is a **new implementation** of the documented mechanism. It is not the
recovered source of historical paper results. New checkpoints, parameter counts,
run times, and accuracies must be measured and reported separately.

## Install and check

Use Python 3.11 or newer, with a PyTorch installation appropriate for your CPU or
CUDA device. From this repository:

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -e ".[vlm,test]"
python -m pytest tests
bbs smoke --output runs/software-smoke --updates 3
```

The smoke command runs CPU training on an explicitly synthetic software fixture.
It verifies collection, replay, optimization, and checkpoint writing; it does not
evaluate a benchmark or reproduce any experimental result. Use a fresh output
directory for each run.

## Implemented workflow

1. Fit the visual projection against **precomputed native alignment targets**.
2. Fit the projection and language query/value LoRA using teacher-forced records.
3. Collect a fixed Grounding replay with the frozen released-input executor.
4. Fit H, CB, or the two Independent controllers from that replay.
5. Evaluate the controller on a separate fixed Grounding manifest.

See **[NAACL runtime guide](docs/NAACL_RUNTIME.md)** for runnable commands, input
schemas, checkpoint semantics, and remaining boundaries. The default settings
are in [`configs/naacl_reference.json`](configs/naacl_reference.json).

Current CLI collection/evaluation supports **Grounding with retrieval disabled**.
Action parsing/aggregation utilities are available as library code; a complete
Action benchmark collection and evaluation command is not supplied. The CLI
saves the final controller checkpoint and does not automate development-set
selection or aggregation over ten families and three evaluation replicates.

## Privacy and experimental scope

The runtime uses 25 regional features, a private probe, independent Gaussian
refinement, and an admission filter with a fixed 56-slot transcript. The
calibration sensitivity `0.01` applies to the stated feature neighborhood, not
arbitrary pairs of screenshots. This is numerical research software, not an
audited production privacy implementation. Do not publish private randomizer
state, trusted replay observations, or clean training features.

For the default new architecture, total controller parameters are **133,706 for
H** and **258,382 for Independent**, including target critics. These are not the
historical manuscript parameter counts; this implementation does not claim an
equal-capacity comparison.

## Earlier implementation

The earlier scripts and packages remain available for inspection. Their original
instructions are preserved in [Legacy usage](docs/LEGACY_USAGE.md). They are not
the entry point for the new `bbs` workflow, and their artifacts should not be
mixed with new runs.
