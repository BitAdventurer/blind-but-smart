# Explicit local dataset preparation

For public downloads and upstream conversion, first use the
[dataset adapters](upstream-data.md). Their `records.jsonl` and `splits.json`
feed this command. Use [training-input preparation](training-inputs.md) next
for projection and adapter fitting.

`bbs prepare-dataset` prepares caller-supplied normalized JSONL records for the
recorded Grounding or Action driver. It does not download datasets, interpret an
unknown upstream benchmark format, infer the manuscript's historical split, or
run models. Convert upstream data to the small interchange contract below using
its documented field meanings before invoking this command.

```powershell
bbs prepare-dataset --records supplied/records.jsonl --splits supplied/splits.json --task G --output data/prepared-G
```

The Python equivalent is
`prepare_dataset(records_path, splits_path, output_dir, task="G", action_schema=None)`
from `gui_joint_control.dataset_preparation`. The output directory must be new.

## Trajectory assignment

The split file is one JSON object containing exactly these three roles:

```json
{"train":["train-001"],"dev":["dev-001"],"test":["test-001"]}
```

Every source trajectory must appear exactly once. Overlapping or duplicate IDs,
unassigned trajectories, unknown assigned IDs and missing roles are errors.
An explicitly empty role is allowed; its empty manifest cannot be executed by
the evaluator until a nonempty population is supplied. No random split is made.
Assignments apply to complete trajectories, so different slots from one
trajectory cannot enter different roles. Train is the fitting/behavior source;
dev is the development-selection source; test is the final evaluation source.
These names identify the caller's supplied populations, not historical paper
populations. A supplied record's optional `split` must match its assigned role.
The recorded driver checks `train` against `--split fit-train`, `dev` against
`--split development`, and `test` against `--split test` or `--split evaluation`
before screen access. Manifests without declarations remain supported with the
number of undeclared records recorded in provenance.

## Grounding records

Each line is one JSON object with a nonempty string `trajectory_id`, an explicit
zero-based integer `slot`, and explicit boolean `eligible`. An eligible Grounding
record also needs public `instruction` (or `thought`, exclusively), an offline
reference box, and exactly one `image_path` or `features_path`:

```json
{"trajectory_id":"train-001","slot":0,"eligible":true,"instruction":"Locate Save","target_box":[0.1,0.2,0.3,0.4],"image_path":"screens/train-001.png","public_metadata":{"source_task_id":"supplied-task-001"}}
```

`target_box` means normalized `[x_min,y_min,x_max,y_max]`, with ordered finite
coordinates in `[0,1]`. For explicit pixel coordinates instead, use
`target_box_pixels` and positive integer `screen_size: [width,height]`:

```json
{"trajectory_id":"train-001","slot":0,"eligible":true,"thought":"Locate Save","target_box_pixels":[20,20,60,40],"screen_size":[200,100],"features_path":"features/train-001.npy"}
```

The second record becomes `instruction: "Locate Save"` and
`target_box: [0.1,0.2,0.3,0.4]`. Coordinates are divided by the explicitly supplied
width and height. The preparer does not clip, reorder, guess units, or infer a
target point from a box. Both box forms or both text forms on one record are
ambiguous and rejected. A supplied optional `task` must equal `G`.

## Action records

```json
{"trajectory_id":"train-001","slot":0,"task":"A","eligible":true,"request":"Save the document","history":["Opened the document","Found the Save menu"],"reference_action":{"function":"caller_defined_save","arguments":{},"status":"caller_defined_ok"},"image_path":"screens/train-001.png","screen_size":[200,100],"reference_boxes":{}}
```

An eligible Action record requires explicit `request`, `history` (possibly `[]`),
and the already canonical `reference_action` object with exactly `function`,
`arguments`, and `status`. Function/status are nonempty strings and arguments is
an object. `INVALID` is reserved for invalid predictions and cannot be a supplied
reference. History is an array of preceding dataset-recorded thought strings in
chronological order. The preparer preserves supplied order; it cannot establish
chronology from bare strings and never constructs history from labels or model
predictions. A supplied optional `task` must equal `A`.

Pass `--action-schema supplied/schema.json` for validation against the existing
frozen schema contract described in [action-retrieval.md](action-retrieval.md).
References must already be canonical: aliases requiring repair are rejected.
Without a schema, preparation checks the object structure only. Recorded Action
evaluation still requires a frozen schema and explicit offline scorer.

Optional `screen_size` is positive integer `[width,height]`. Optional
`reference_boxes` is an offline mapping preserved verbatim, with no guessed
coordinate conversion. Supply the representation and coordinate units required
by the pinned scorer. When that scorer expects normalized boxes, callers must
supply normalized boxes; this preparer does not infer normalization for Action.
Canonical spatial Action arguments must follow the frozen schema's normalized
coordinate contract. Labels, boxes, and screen size are offline scoring inputs;
the recorded driver passes only released features, public request and preceding
thoughts to the executor.

## Preservation, validation and outputs

Slots keep their supplied indices. Ineligible records stay in their assigned
manifest and counts; they need no text, label, or asset. Supplied optional fields
are validated when present. Other per-record metadata, including
`public_metadata`, is preserved. Missing slots remain missing, allowing the
runtime to mark them unrecorded rather than renumbering later slots.

Image/feature paths resolve relative to the source JSONL's directory and are
rebased relative to the output directory. Eligible assets must exist as local
files. Preparation does not decode assets, inspect feature-array shape, or hash
their private contents. The runtime validates/loads them on admitted execution.

Optional `native_target_tokens_path` for later alignment preparation is also
validated and rebased. It is a trusted training artifact, not an executor input.

Each trajectory exceeding 56 recorded slots, or containing an index beyond 55,
is excluded in its entirety. It is never truncated or split into smaller
trajectories. `exclusions.jsonl` records its ID, assigned role, record count,
maximum slot, and reason. Duplicate trajectory/slot keys, negative/boolean
indices, conflicting roles/tasks, invalid boxes, missing mandatory fields,
duplicate JSON keys and nonfinite JSON values are errors. Malformed input aborts
the complete request before creating output. Existing output is refused. Once
validation finishes, complete output is written in a staging directory and
renamed into place, while source files remain intact.

The output contains `train.jsonl`, `dev.jsonl`, `test.jsonl`,
`exclusions.jsonl`, and `preparation.json`. The report binds source records and
split-assignment bytes by SHA-256, records each generated manifest digest,
optional schema digest, supplied totals, and per-role accepted/excluded
trajectories and records plus eligible/ineligible counts. These are preparation
counts for the supplied files, not reproduced experimental results. The output
manifests and provenance are trusted local artifacts containing reference labels.
