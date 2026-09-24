# Action and frozen retrieval artifact contracts

The current manuscript's Action path is implemented from released latents to
candidate generation, strict parsing, relevance, sampled-coordinate aggregation,
protected feedback, and recorded-screen replay. This is new reference software,
not evidence that the reported Action measurements have been rerun.

## Action schema and evaluator

`load_action_schemas(path)` requires an explicitly supplied JSON artifact:

```json
{
  "schema_id": "caller-pinned-version",
  "source_split": "fit-train",
  "source_manifest_sha256": "<64 hexadecimal characters>",
  "functions": {
    "caller_defined_function": {
      "required": ["point"],
      "optional": ["text"],
      "spatial": ["point"],
      "statuses": ["caller_defined_status"],
      "argument_types": {"text": "string"},
      "status_aliases": {}
    }
  },
  "function_aliases": {}
}
```

The names above illustrate the schema format only; they do not claim the official
GUI-360 vocabulary. Supply its pinned function/status vocabulary and Train-only
canonicalization map. Supported argument type declarations are `string`,
`integer`, `number`, `boolean`, `array`, `object`, and `null`. Spatial values are
normalized numeric two-element arrays; booleans are not coordinates. Values
outside `[-1e-6,1+1e-6]` are invalid; only boundary roundoff is clipped.

The offline evaluator is supplied explicitly. Its callable signature is:

```python
def evaluate(action, slot):
    # Import and call your pinned official evaluator here. It can access
    # slot.reference_action, slot.reference_boxes and slot.screen_size.
    # Return the official component scores as Python bool values:
    return ActionScores(function_correct, arguments_correct, status_correct)
```

The code does not invent a replacement official evaluator or a default semantic
normalizer. Bind the official box-containment / 25-pixel fallback behavior in that
adapter. The run records the adapter source digest; dependencies of a separately
installed official scorer also need an immutable package/revision binding in the
run's environment record. Invalid predictions and exhausted eligible slots receive
zero on all components without calling the evaluator. Exact-step correctness is
the conjunction of the three components.

`ActionSlot` carries the public request, chronological dataset-recorded thought
history, offline reference action, fixed eligibility, and optional offline scoring
metadata. `run_action_trajectory` passes only `(release, request, history, k,
prior_feedback)` to the executor. No reference action, box, screen size, correctness
label, or future reference thought is passed to the allocator or executor. The
prompt retains at most ten preceding thoughts, then removes oldest history before
lowest-ranked retrieved demonstrations if the complete input exceeds 4,096 tokens.
Current request text is never truncated or Unicode-normalized.

`ReleasedQwenExecutor.predict_action` runs one batch over the same completed
release for every candidate, with at most 128 output tokens. It returns
`(action_object, protected_feedback, candidates)`. Generated candidate text and
token IDs are exported alongside likelihood and release-only relevance. Truncated
or malformed outputs are invalid; they are not repaired using a reference action.

## Frozen demonstration bank

`FrozenDemonstrationBank.from_files(manifest_json, keys_npy, task=..., view=..., schemas=...)`
accepts an externally prepared bank declaration and NumPy key artifact. It does
not create demonstrations from evaluation predictions or update a bank online.
The key artifact stores one 256-dimensional **mean clean fit-train feature** per
source row, before normalization. The loader applies the `1e-12` norm floor.

Required manifest fields:

- `task`: `G` or `A`, with `source_split: "fit-train"`.
- `source_manifest_sha256`, `evaluation_manifest_sha256`, `keys_sha256`:
  immutable artifact bindings. The key-file digest is checked on load. The caller
  must also bind the complete evaluation manifest used for exclusions to the run.
- `success_binding`: `source_field`, `granularity: "trajectory"`, and
  `missing_policy: "error"` or `"exclude"`. This records the author's supplied
  upstream success convention, rather than guessing one from the manuscript.
- `evaluation_public_records`: all public metadata records of the complete
  evaluation population against which the bank view is frozen.
- `entries`: source rows in the same order as key rows. Each row has
  `trajectory_id`, `original_slot`, `task`, `source_split`, boolean `eligible`,
  boolean `schema_valid`, boolean `trajectory_success`, `public_metadata`, and
  a typed `value`.

Public metadata requires nonempty `source_task_id`, `template_id`, `document_id`,
and `normalized_instruction`. Strict views additionally require `template_family`
and `document_family`. Exact matches against any evaluation record exclude a bank
row globally. Strict views also exclude matching families and normalized-instruction
token Jaccard similarity at least 0.80. Missing identities abort construction;
neither evaluation images nor clean features are used for membership decisions.

The bank first admits eligible, schema-valid rows with successful upstream
trajectory labels, sorts by trajectory and original slot, retains the last 10,000
rows under FIFO, and then applies the fixed exclusion view. A declared missing
success label is excluded only when the explicit missing-value policy says so.
The loader validates these declarations and hashes; it cannot independently
authenticate historical provenance.

Before a run, `bank.validate_evaluation_manifest(path)` checks the actual complete
JSONL artifact's digest and the multiset of its task-matching `public_metadata`
against the bank declaration. A combined development/test exclusion manifest is
permitted when the driver also verifies that every current run identity and its
public metadata belong to that combined manifest. This binds a global view,
rather than filtering bank membership separately for individual queries.

Grounding values contain exactly `instruction`, `coordinate` (normalized pair),
and `element_type`. Action values contain exactly `instruction`,
`reference_action` (canonical function/arguments/status object), and `schema_id`.
Action bank construction additionally requires the explicit loaded schema map;
each reference must already parse to that same canonical tuple.
No images, clean features, success labels, or untyped free text are serialized
into the prompt. Retrieval text uses NFC, sorted object keys, compact UTF-8 JSON,
and Python 3's shortest-round-trip floating-point serialization; record the Python
version with the run. Original current instruction/request strings retain their
separate non-normalization contract.

`bank.retrieve(release, prior_feedback, threshold)` gates on strict
`prior_feedback > threshold`, where the explicit development-selected threshold
belongs to `{0,0.25,...,2}`. Query keys use only the mean completed refinement
release. Scores must exceed 0.15; at most eight are returned by decreasing score,
then trajectory ID, original slot, and task. Zero keys/query means yield zero
similarity. The strict view uses the primary-selected threshold. Single-screen
protocols must pass an empty/disabled bank.

## Failed runs and evidence boundary

Prompt preflight and Action reference validation should finish before any private
screen access. If generation, parsing infrastructure, or the externally bound
scorer raises after a release, `TrajectoryExecutionError` retains the partial
records and already-spent budget. The driver must withhold aggregate benchmark
results for that failed run. Ordinary candidate parse failures instead retain
their original invalid-output treatment and remain in the denominator.

Saved replay can preserve `slot_id`, `next_slot_id`, `source_manifest_sha256`,
`task`, and `family_id` as non-pickle Unicode arrays. When a metadata field is
present it must be present on every transition; paired current/next slot IDs must
both be supplied. The terminal next-slot ID is the empty string. These public
bindings allow development disjointness and TMS counterpart checks without
injecting target labels into the allocator.

The synthetic tests exercise access boundaries, repeated release reuse, strict
schema handling, aggregation, fixed denominators, threshold gates, immutable bank
keys, global exclusion views, and hash mismatch rejection. They do not measure
Grounding/Action accuracy or establish the historical split, schema, bank, or
checkpoint provenance.
