# Recorded run summaries

```bash
bbs summarize-run --run-directory runs/H-test --output reports/H-test.json
```

`gui_joint_control.reporting.summarize_run(run_dir, output_path)` creates a new
JSON report from an existing `run.json` and `transcript.jsonl` pair.
`build_summary(run_dir)` returns the same report without writing it. Neither
function runs a model or changes the source records; an existing output is an
error. Supported kinds are `new_recorded_controller_evaluation` and
`new_behavior_collection`. Training reports, aborted runs, and the separate
selected-development filenames are not accepted.

```bash
bbs summarize-run --run-directory runs/supplied-evaluation --output reports/run-summary.json
```

The exporter verifies the transcript's recorded SHA256, matching row identities
when supplied, the complete 56-slot grid for each trajectory, boolean outcome
labels, and cumulative debit/remaining-budget arithmetic. Recorded counts and
accuracy must agree with the transcript. Older transcripts do not record task,
family, replicate, or controller method on each row; the report explicitly
lists these omissions instead of inventing a row-level binding.
Prepared manifest roles are checked against the requested execution split before
release; summaries preserve that check and count records with no declared role.

Each report aggregates one supplied run. With `N` eligible slots, `I`
invocations, `C` correct outcomes, `K` generated candidates, and `D` the sum
of executed regional budgets:

| Report metric | Calculation |
| --- | --- |
| `accuracy` | `C / N` |
| `accuracy_per_invocation` | `C / I` |
| `invocation_rate` | `I / N` |
| `mean_regional_budget_per_eligible` | `D / (25 * N)` |
| `mean_regional_budget_per_invocation` | `D / (25 * I)` |
| `candidates_per_invocation` | `K / I` |
| `candidates_per_eligible` | `K / N` |
| `budget_cap_fraction` | `D / (75 * N)` |

Exhausted eligible slots remain failures in `N`. Task and structural padding
are counted in the fixed grid, with zero candidates and zero additional debit,
but do not enter `N`. An empty denominator yields JSON `null`. Action component
accuracy uses the same eligible denominator. Status counts and proportions are
reported separately per fixed slot and per eligible slot.

The report links these aggregates to the actual source JSON/transcript hashes,
recorded artifact bindings, model/tokenizer revisions, prompt-policy hashes and
settings, and safe runtime/device metadata. Missing recorded fields are `null`
or explicitly listed. Local snapshot identifiers and paths, raw prompts,
predictions, candidate exports, replay records, admitted-input hashes, private
randomness, and unrecognized nested provenance fields are excluded.

Artifact bindings are opaque recorded identifiers. The exporter verifies the
supplied transcript, not the current contents of model/checkpoint/manifest files
elsewhere. It does not infer original experiment settings, combine fitting
families or replicates, calculate uncertainty, or reproduce historical results.
These offline diagnostic aggregates are not certified differentially private outputs.
