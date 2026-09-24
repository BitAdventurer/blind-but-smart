# Development selection and explicit family plans

The current profile separates `selected.pt` (development-selected) from `last.pt`
(latest resumable optimizer state). Neither file is a recovered historical
checkpoint. New commands produce new runs; this machinery does not reproduce
the manuscript's reported values by itself.

## Selection

`gui_joint_control.selection.CheckpointManager` evaluates every 10,000 completed
component iterations. Its callback runs actual development trajectories using
the candidate controller's own feedback and ledger, then returns:

```python
{
    "split": "development",
    "manifest_sha256": actual_development_manifest_sha256,
    "episodes": [
        {"trajectory_id": actual_trajectory_id, "replicate": registered_replicate,
         "records": actual_56_slot_runtime_records},
        # Every registered trajectory and replicate.
    ],
}
```

Every record has `slot`, `eligible`, `invoked`, `correct`, `executed_budgets`
and `candidate_count`, as emitted by the runtime. Return uses the eligible-slot
clock, discount 0.99, the correctness/resource reward for invoked slots, and -1
for each exhausted eligible slot. Ineligible slots do not advance this clock.
The manager recomputes invocation reward; it does not add the replay exhaustion
penalty a second time and does not include training entropy. Returns are first
averaged equally over eligible trajectories, then over registered replicates.
Resource tie-breaks use total expenditure divided by 25 times the eligible-slot
count, then total candidates divided by eligible slots; the final tie favors the
earlier iteration. Empty eligible development populations are invalid.

The callback must return the same trajectory/replicate/eligibility grid at every
candidate checkpoint. A test/Bench split, changed manifest, missing scheduled
checkpoint, or changed grid fails selection. The Independent trainer evaluates
its composed pair at candidate checkpoints without joint updates; both component
states are saved together. This is the explicit new implementation's checkpoint
selection convention, not verification of a historical selection record.

Typical API use:

```python
manager = CheckpointManager(output_directory, binding=run_binding)
# run_binding includes dev_manifest_sha256 plus task/family/config/executor IDs.
for _ in range(additional_updates):
    trainer.step()
    manager.consider(trainer, development_evaluator)
manager.save_last(trainer)
```

If fewer than 10,000 iterations have completed, `last.pt` can exist while
`selected.pt` does not. A latest checkpoint is never silently described as
development-selected. `selection.json` binds the best score, evaluated grid,
iterations, seed/config metadata, and hashes of selected/last checkpoint files.
Resume in the same output directory with `resume=True`, load `last.pt`, and call
`validate_resume(trainer)` before the next step. Corrupt hashes, wrong trainer
state, and interrupted selection boundaries fail rather than dropping the best
checkpoint or silently restarting the selection history.

`select_retrieval_threshold(callback, manifest_sha256=...)` evaluates all nine
thresholds 0, 0.25, ..., 2 on the same development grid. It maximizes the same
return and resolves equal returns in favor of the larger threshold. Its callback
must run the supplied threshold; the helper supplies no fabricated scores.

## Ten-family registry

`build_plan(registry, output_root)` validates inputs and returns argv jobs without
executing them. The CLI's protocol preparation command is also read-only with
respect to models. `execute_plan(plan)` is a separate explicit action and can
perform substantial fitting/evaluation; do not call it merely to inspect a plan.

This plan begins with **existing frozen executor artifacts**. It orchestrates
controller fitting followed by three evaluations of each selected controller.
It does not perform or establish ten complete projection/adapter refits. Bind
the correctly fitted executor for each family/task before making that claim.

Required registry shape (the names below are placeholders, not example seeds
or measured artifacts; incomplete placeholders are rejected):

```text
schema_version: 1
families: exactly 10 entries
  family_id: explicit unique ID
  public_root: explicit unique 64-character lowercase hex commitment
  tasks: one or both G and A, same population in every family
    task: G or A
    model: frozen model repository ID or absolute local model path
    revision: immutable 40-character lowercase hex model revision
    model_artifact: {path, sha256}, additionally required for local models
    config: {absolute path, sha256}
    replay: {absolute path, sha256}
    dev_manifest: {absolute path, sha256}
    manifest: {absolute path, sha256} for final evaluation
    projection: {absolute path, sha256}
    paired_h_checkpoint: {absolute path, sha256}, required for TMS-based baselines
    dev_replicates: positive integer (default 3, shared across methods)
    executor_flags: explicit CLI flag/value bindings described below
    executor_artifacts: CLI flag -> {absolute path, sha256}
    methods: same trainable-method population in every family/task
      method: H, CB, Disclosure-only, Count-only, Independent, Independent-1M
      seed: explicit controller seed
      fit_output: absolute new output directory
      fit_argv: complete string array for the train command
      schedule_artifacts: CLI flag -> {absolute path, sha256}, when applicable
      evaluations: exactly 3 entries
        replicate_id: explicit ID
        seed: explicit public decoding seed
        output: absolute new output directory
        argv: complete string array for collect-grounding or collect-action
```

All artifacts are checked before plan creation and again before commands run.
Development and final evaluation manifests must be shared across all fitting
families for each task. The frozen retrieval bank, keys, exclusion manifest and
view are likewise shared per task; development-selected gate thresholds may
differ by family. Fitted executor/projection artifacts may differ by family.
For a directory, `artifact_digest(path)` hashes the canonical sorted list of
`[relative POSIX path, file SHA-256]` pairs over every file. A remote revision
flag does not make mutable local weights immutable. Local tokenizer and DINO
directories likewise require `tokenizer_artifact` and `dino_model_artifact`;
remote tokenizer/DINO IDs require their immutable revision flags.

`executor_flags` supports `--dtype`, `--device`, `--tokenizer`,
`--tokenizer-revision`, `--dino-model`, `--dino-revision`,
`--retrieval-threshold`, `--retrieval-view`, `--action-evaluator`, and the boolean
`--disable-retrieval`. `executor_artifacts` supports `--public-projection`,
`--action-schema`, `--retrieval-bank`, `--retrieval-keys`, and
`--retrieval-exclusion-manifest`. Retrieval requires a bound bank, keys and
selected threshold; `--disable-retrieval: true` is an explicit ablation and
must not be presented as the full retrieved method. Action additionally requires
the typed schema, evaluator entry point, and `action_evaluator_source` artifact.
Record the scorer's complete dependency environment alongside its source hash.

Standalone Disclosure-only and Count-only require all three `schedule_artifacts`:
`--tms-schedule` for replay training, `--dev-tms-schedule` for development and
`--evaluation-tms-schedule` for final evaluation. Independent and Independent-1M
require only `--tms-schedule`: their learned heads are composed during development
and final evaluation, so no fixed complementary evaluation schedule is needed.
Required schedules must be constructed from
the paired H development traces for the intended eligible population; arbitrary
constant complements do not satisfy this protocol. Commands use the training
schedule during checkpoint loading and, for standalone single-head controllers,
the split-specific schedule during rollout. Supplying a schedule flag that is
absent from the registry is rejected.

Plan these dependencies in two stages: first fit/select H (optionally together
with CB), then construct schedules from that selected H's actual development
traces, then prepare the TMS-dependent baseline plan. A plan that both refits H
and consumes pre-existing TMS schedules is rejected: those means could belong to
another H checkpoint. Baseline tasks explicitly supply `paired_h_checkpoint`;
every schedule's task, family, selected-H hash and development-manifest hash must
match it. This staging preserves the pairing without pretending that preparation
has already executed a future H fit or produced its development measurements.

Each command must start with `bbs` or `python -m gui_joint_control.cli`; shell
command strings and duplicate flags are rejected. The planner checks exact
method, updates, replay, configuration, development/final manifest, frozen
executor, TMS schedules, family, task, seed and output bindings. Fitting uses
1,000,000 updates per component except baseline Independent's 500,000, and batch
size 256. All three final evaluations must load `fit_output/selected.pt`; `last.pt`
cannot substitute for it. The execution journal pins the selected file hash
across those replicates. Commands run without a shell, and a failed or partial
execution is never silently retried.

Controller seeds must be distinct across family/task/method runs. Replicate IDs
and public decoder seeds are paired across methods within a family/task, with
three distinct decoder seeds. These are newly supplied explicit bindings; the
planner does not pretend to recover or automatically implement the historical
CAS-v16 seed registry. Public roots and these seeds never seed secret probe or
refinement innovations. Runtime must generate independent secret streams.

Synthetic tests exercise arithmetic, tie-breaking, state resume, plan validation
and command ordering through fake callbacks only. Their success is software
verification, not a benchmark measurement or evidence of ten executed refits.
