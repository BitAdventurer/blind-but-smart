"""Synthetic completed records only; no model inference or benchmark claims."""
from hashlib import sha256
import json

import pytest

from gui_joint_control.reporting import build_summary, summarize_run


def write_run(path, *, task="G", identities=True):
    path.mkdir()
    identity = {"task": task, "family_id": "family-1", "replicate_id": "2", "controller_method": "H"}
    rows = []
    used = 0.0
    for slot in range(56):
        invoked = slot < 3
        budgets = [5.] * 25 if slot < 2 else [1.5] * 25 if invoked else None
        if invoked:
            used += sum(budgets)
        status = ("EXEC_PROP" if slot < 2 else "EXEC_FALLBACK" if slot == 2
                  else "FILTER_EXHAUSTED" if slot == 3 else "TASK_PAD" if slot == 4 else "STRUCTURAL_PAD")
        row = {"trajectory_id": "private-screen-episode", "slot": slot,
               "eligible": slot < 4, "invoked": invoked, "correct": slot in (0, 2),
               "status": status, "candidate_count": slot + 2 if invoked else 0,
               "executed_budgets": budgets, "used_budget": used, "remaining_budget": 300 - used,
               "point": [.5, .5], "instruction": "private prompt", "private_rng_root": "secret"}
        if identities:
            row.update(identity)
        if task == "A":
            row.update(function_correct=invoked, arguments_correct=row["correct"], status_correct=invoked,
                       action={"function": "private-action", "arguments": {"text": "secret"}})
        rows.append(row)
    run = {"kind": "new_recorded_controller_evaluation", "reproduces_historical_results": False,
           **identity, "split": "test", "eligible": 4, "invoked": 3, "correct": 2, "accuracy": .5,
           "manifest_split": {"requested_split": "test", "declared_roles": ["test"], "undeclared_records": 0},
           "executor": {"model": "C:/private/model", "model_revision": "a" * 40,
                        "tokenizer_revision": "b" * 40, "private_rng_root": "secret"},
           "model_snapshot": {"identifier": "C:/private/model", "tree_sha256": "c" * 64},
           "tokenizer_snapshot": {"identifier": "C:/private/tokenizer", "revision": "b" * 40},
           "manifest_sha256": "d" * 64, "controller_checkpoint_sha256": "e" * 64,
           "controller_config_sha256": "f" * 64,
           "prompt_policy": {"version": "jdc-reference-v1", "token_limit": 4096,
                             "max_previous_steps": 10, "renderer_sha256": "1" * 64,
                             "policy_sha256": "2" * 64, "private_prompt": "secret"},
           "private_rng_root": "secret", "trusted_input_sha256": "private-hash",
           "replay": {"records": [{"clean_features": "secret"}]}, "output": "C:/private/output"}
    if task == "A":
        run.update(function_correct=3, arguments_correct=2, status_correct=3,
                   function_accuracy=.75, arguments_accuracy=.5, status_accuracy=.75)
    save_run(path, run, rows)
    return run, rows


def save_run(path, run, rows):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (path / "transcript.jsonl").write_text(text, encoding="utf-8")
    run["transcript_sha256"] = sha256((path / "transcript.jsonl").read_bytes()).hexdigest()
    (path / "run.json").write_text(json.dumps(run), encoding="utf-8")


@pytest.mark.parametrize("task", ["G", "A"])
def test_summary_preserves_exhausted_denominators_and_exports_only_aggregates(tmp_path, task):
    source = tmp_path / "source"
    write_run(source, task=task)
    original = {file.name: file.read_bytes() for file in source.iterdir()}
    output = tmp_path / "summary.json"
    report = summarize_run(source, output)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"] == {"trajectories": 1, "fixed_slots": 56, "eligible": 4,
                                "invoked": 3, "correct": 2, "candidates": 9}
    metrics = report["metrics"]
    assert metrics["accuracy"] == .5 and metrics["invocation_rate"] == .75
    assert metrics["candidates_per_invocation"] == 3 and metrics["candidates_per_eligible"] == 2.25
    assert metrics["regional_debit"] == 287.5
    assert metrics["mean_regional_budget_per_eligible"] == 2.875
    assert metrics["mean_regional_budget_per_invocation"] == pytest.approx(287.5 / 75)
    assert metrics["budget_cap_fraction"] == pytest.approx(287.5 / 300)
    assert report["status_proportions"]["per_eligible_slot"]["FILTER_EXHAUSTED"] == .25
    assert report["status_proportions"]["per_fixed_slot"]["STRUCTURAL_PAD"] == 51 / 56
    assert report["provenance"]["executor"]["model_revision"] == "a" * 40
    assert report["provenance"]["artifact_bindings"]["manifest_sha256"] == "d" * 64
    assert report["provenance"]["prompt_policy"]["policy_sha256"] == "2" * 64
    assert not report["missing_information"]
    exported = json.dumps(report)
    assert all(text not in exported for text in ("private", "secret", '"instruction"', '"point"', "clean_features"))
    assert {file.name: file.read_bytes() for file in source.iterdir()} == original
    with pytest.raises(FileExistsError):
        summarize_run(source, output)
    if task == "A":
        assert metrics["function_accuracy"] == metrics["status_accuracy"] == .75
        assert metrics["arguments_accuracy"] == .5


def test_legacy_identity_omissions_are_explicit_and_empty_denominators_are_null(tmp_path):
    source = tmp_path / "source"
    run, rows = write_run(source, identities=False)
    run.pop('manifest_split')
    save_run(source, run, rows)
    legacy = build_summary(source)
    assert "transcript.controller_method: absent in 56 rows" in legacy["missing_information"]
    assert 'manifest_split' in legacy['missing_information']
    for row in rows:
        row.update(eligible=False, invoked=False, correct=False, status="STRUCTURAL_PAD",
                   candidate_count=0, executed_budgets=None, used_budget=0, remaining_budget=0)
    run.update(eligible=0, invoked=0, correct=0, accuracy=None)
    save_run(source, run, rows)
    metrics = build_summary(source)["metrics"]
    assert metrics["regional_debit"] == 0
    assert all(value is None for key, value in metrics.items() if key != "regional_debit")


def test_export_rejects_changed_transcript_without_creating_output(tmp_path):
    source = tmp_path / "source"
    write_run(source)
    with (source / "transcript.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="SHA256"):
        summarize_run(source, output)
    assert not output.exists()


def test_actual_cuda_device_index_and_safe_environment_survive_without_host_identity(tmp_path):
    source = tmp_path / "source"
    run, rows = write_run(source)
    run.update(model_device="cuda:0", model_dtype="bfloat16",
               runtime={"device": "cuda:0", "dtype": "bfloat16", "cudnn": 91002,
                        "gpu_name": "NVIDIA GeForce RTX 5090", "os": "Windows", "os_release": "11",
                        "hostname": "private-host", "user": "private-user"})
    save_run(source, run, rows)
    provenance = build_summary(source)["provenance"]
    assert provenance["model_device"] == provenance["runtime"]["device"] == "cuda:0"
    assert provenance["runtime"]["cudnn"] == 91002
    assert provenance["runtime"]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert "private" not in json.dumps(provenance)
    run["runtime"]["device"] = "C:/private/device"
    save_run(source, run, rows)
    with pytest.raises(ValueError, match="device must"):
        build_summary(source)


@pytest.mark.parametrize("change,match", [
    ("missing_slot", "complete 56-slot"),
    ("duplicate_slot", "Duplicate"),
    ("identity", "family_id differs"),
    ("count", "Recorded eligible"),
    ("padded_correct", "Correctness requires"),
    ("debit", "cumulative budget"),
    ("early_exhaustion", "reserve is exhausted"),
    ("aborted", "aborted"),
])
def test_even_rebound_transcripts_cannot_change_denominators_or_ledger(tmp_path, change, match):
    source = tmp_path / "source"
    run, rows = write_run(source)
    if change == "missing_slot":
        rows.pop()
    elif change == "duplicate_slot":
        rows.append(rows[0])
    elif change == "identity":
        rows[0]["family_id"] = "other-family"
    elif change == "count":
        run["eligible"] = 3
    elif change == "padded_correct":
        rows[3]["correct"] = True
    elif change == "debit":
        rows[3]["used_budget"] -= 1
    elif change == "early_exhaustion":
        rows[0].update(invoked=False, correct=False, candidate_count=0,
                       status="FILTER_EXHAUSTED", executed_budgets=None)
    elif change == "aborted":
        (source / "aborted.json").write_text("{}", encoding="utf-8")
    save_run(source, run, rows)
    with pytest.raises(ValueError, match=match):
        summarize_run(source, tmp_path / "summary.json")
    assert not (tmp_path / "summary.json").exists()
