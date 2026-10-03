# Build projection and adapter inputs

`prepare-alignment` and `prepare-supervised` build inputs for `fit-projection`
and `fit-adapter`; they do not optimize models. Supply explicit `train` or
`fit-train` manifests from [data preparation](dataset-preparation.md).
Development/test rows and duplicate IDs are rejected. Only eligible rows enter
training artifacts. Source/model bindings and output hashes go in
`preparation.json`. Clean features and targets remain trusted-side artifacts.

Local models are the default. Remote model/tokenizer/DINO identifiers require
an immutable 40-hex revision and `--allow-download`. Keep snapshots in `models/`.

## Regional image features

For `image_path` inputs supply DINO and the saved public projection. Generate
the existing reference matrix once, then reuse it at fitting and evaluation:

```bash
python -c "import numpy as np; from gui_joint_control.features import public_projection; np.save('data/public-projection.npy', public_projection())"
```

This public DINO-to-256 matrix is distinct from the learned Qwen projection.
For precomputed, unit-clipped `[25,256]` `features_path` arrays, omit
DINO/projection arguments.

## Stage 1: native targets

```bash
bbs prepare-alignment --manifest data/prepared-G/train.jsonl --model models/qwen-base --dino-model models/dinov2-base --public-projection data/public-projection.npy --device cuda --dtype bfloat16 --output data/alignment-G
bbs fit-projection --records data/alignment-G/alignment.npz --device cuda --output runs/projection-G
```

The native encoder follows the manuscript's whole-screen resize and pinned
processor normalization, validating grid `[1,10,10]`, merge size two and exactly
25 tokens. It does not infer crop pooling. Alternatively provide per-row
`native_target_tokens_path` pointing to numeric `[25,H]` NPY arrays and omit
`--model`. Precomputed features plus targets need no model inference.

`alignment.npz` contains `features`, `native_target_tokens`, and `record_ids` in
the recorded order. Target dimensions must agree across rows.

## Stage 2: teacher-forcing records

```bash
bbs prepare-supervised --manifest data/prepared-G/train.jsonl --task G --tokenizer models/qwen-base --dino-model models/dinov2-base --public-projection data/public-projection.npy --device cuda --output data/supervised-G
bbs fit-adapter --model models/qwen-base --projection runs/projection-G/projection.npy --records data/supervised-G/records.jsonl --device cuda --dtype bfloat16 --save-merged --output runs/adapter-G
```

Grounding requires explicit normalized coordinate `target_text`, for example
`{"x":0.2,"y":0.4}`. The GUI-360 converter preserves supplied raw coordinates;
no box-center target is invented. Action uses canonical `reference_action`;
pass `--task A --action-schema supplied/schema.json`.

The builder reuses the evaluation prompt renderer without retrieval, preserving
current text and retained preceding Action thoughts. It validates one group of
25 image-pad tokens and vision delimiters. Labels are tokenized separately and
end in one tokenizer EOS (`--eos-token-id` may select from the tokenizer's
declared EOS set, never introduce an unrelated token).
Labels never enter the prompt. Interior EOS, padding/visual target tokens,
malformed targets and overlong mandatory prompts are rejected.

Output contains `records.jsonl`, local `features/*.npy` and `preparation.json`.
It feeds `fit-adapter` directly. Both fitting commands validate a present
preparation sidecar before model loading/optimization, including input/feature
hashes and IDs, and retain its hash in fitting provenance. Legacy supplied
inputs without a sidecar remain supported with an explicit null preparation
binding. These are new reference preparations rather than reconstructed
historical fitting artifacts.
