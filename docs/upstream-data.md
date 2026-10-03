# Download and convert official datasets

Install `python -m pip install -e ".[data,vlm,test]"`. Commands use new output
directories; installation never downloads data or launches experiments.

| Dataset | Official source | Example pinned revision |
|---|---|---|
| ScreenSpot-v2 | [OS-Copilot/ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) | `5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c` |
| GUI-360 | [vyokky/GUI-360](https://huggingface.co/datasets/vyokky/GUI-360) | `1c46362efaae836c601004e77b98a1aba89a530f` |

These inspected snapshots are examples for new runs, not recovered paper split
bindings. Downloads require a full commit SHA, validate selected file patterns,
and record the repository/revision/file list in `download.json`. Upstream terms
and licenses apply. Keep downloaded data and generated inputs in ignored `data/`.

## ScreenSpot-v2

```bash
bbs download-dataset --dataset screenspot-v2 --revision 5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c --output data/screenspot-source
bbs extract-images --archive data/screenspot-source/screenspotv2_image.zip --output data/screenspot-unpacked
bbs convert-screenspot --annotations data/screenspot-source/screenspot_desktop_v2.json data/screenspot-source/screenspot_mobile_v2.json data/screenspot-source/screenspot_web_v2.json --images data/screenspot-unpacked/screenspotv2_image --output data/screenspot-converted
bbs prepare-dataset --records data/screenspot-converted/records.jsonl --splits data/screenspot-converted/splits.json --task G --output data/screenspot-prepared
```

Set `--images` to the actual extracted folder containing the image filenames
listed by the JSON (adjust the example's ZIP top-level folder if necessary).
ZIP extraction checks member destinations before writing. Conversion uses native
image dimensions to change pixel `[x,y,width,height]` to normalized xyxy boxes.
Every row, including duplicates, remains a separate one-slot test episode.
Instructions remain verbatim. ScreenSpot never supplies train/dev examples.

## GUI-360 raw trajectories

Choose explicit file patterns; this example selects one evaluation domain:

```bash
bbs download-dataset --dataset gui360 --revision 1c46362efaae836c601004e77b98a1aba89a530f --include "test/data/excel/in_app/success/*.jsonl" --include "test/image/excel/in_app/success/*" --output data/gui360-source
bbs convert-gui360 --root data/gui360-source/test --task G --role test --output data/gui360-G-converted
bbs prepare-dataset --records data/gui360-G-converted/records.jsonl --splits data/gui360-G-converted/splits.json --task G --output data/gui360-G-prepared
```

`--root` contains `data/<domain>/<category>/<success-or-fail>/*.jsonl` and matching
`image/` paths. Download both records and referenced clean images. Grounding uses
current `step.thought`, `action.rectangle`, and explicit `coordinate_x/y` for
normalized training `target_text` when available; it never substitutes box centers.
Accessibility trees and annotated images do not enter the generated records.

Complete ordered one-based `step_id` becomes a zero-based slot. Trajectories
over 56 steps are excluded whole. Ineligible task rows remain at their original
slots; malformed required records and missing eligible images cause errors.
`conversion.json` records source hashes/rules/counts; `exclusions.jsonl` records
whole-trajectory exclusions. Output `records.jsonl` and `splits.json` feed the
existing preparer.

For training, use official `train/` files and `--role train`, then explicitly
assign whole trajectories to train/dev in `splits.json` before preparation.
There is no random or inferred development split. Do not relabel official Test
or ScreenSpot as training data. Raw conversion is distinct from the manuscript's
processed inventories/raw-ID joins and does not automatically yield table Ns.

## Action fields and scorer boundary

Action conversion requires a [frozen schema](action-retrieval.md) and argument map:

```bash
bbs convert-gui360 --root data/gui360-source/test --task A --role test --action-schema supplied/schema.json --action-map supplied/gui360-action-map.json --output data/gui360-A-converted
```

Example map for a schema declaring spatial `point` and optional `button`/`double`:

```json
{"click":{"arguments":{"point":["action.coordinate_x","action.coordinate_y"],"button":"args.button","double":"args.double"},"boxes":{"point":"rectangle"}}}
```

`action.NAME` reads an action-level field; `args.NAME` (or bare `NAME`) reads
`action.args`. A two-key list creates a pair. Schema-declared spatial pairs are
normalized by image dimensions. Explicit maps avoid guessing function vocabularies.
The official top-level coordinates take precedence when the map requests them.

Action inputs are the request and preceding raw thoughts. Upstream status mapping
is `OVERALL_FINISH -> FINISH`, `FINISH -> CONTINUE`. A nonempty API function
excludes the whole trajectory; an empty API-labelled terminal alone does not.
Unsupported functions/task tags retain ineligible slots. Mapped targets must
validate against the frozen schema. Missing required mapping inputs are errors.

Reference boxes are normalized xyxy, preserving parseable off-screen bounds.
Malformed optional boxes are omitted and recorded while keeping the eligible
target for the bound official scorer's point-distance fallback. The adapter does
not replace the official Action scorer.

Field meanings were checked against the [official raw schema](https://huggingface.co/datasets/vyokky/GUI-360/blob/1c46362efaae836c601004e77b98a1aba89a530f/README.md)
and [upstream evaluator](https://github.com/2020-qqtcg/GUI-360/blob/a9f9d2e6f125c8cbc176b46a74aeef76ed16f0f6/evaluator/action_prediction.py).
