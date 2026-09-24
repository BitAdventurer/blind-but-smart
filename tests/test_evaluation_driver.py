"""End-to-end synthetic driver checks; no pretrained weights or dataset runs."""
import argparse
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gui_joint_control.action_evaluation import ActionScores
from gui_joint_control.cli import add_executor_arguments
from gui_joint_control.evaluation import RecordedEvaluator, file_hash, slot_identifier
from gui_joint_control.replay import ReplayBuffer
from gui_joint_control.scoring import ActionCandidate, GroundingCandidate, GroundingResult


class FakeModel:
    """A deterministic software fixture with the same released-input boundary."""
    provenance = {"kind": "synthetic_driver_fixture"}
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [ord(c) + 1 for c in text])

    def __init__(self):
        self.calls = []
        self.prompts = []

    def prompt_token_ids(self, text):
        self.prompts.append(text)
        # Character tokens are a fixture, never a production token estimator.
        return np.asarray([[1] * (27 + len(text))], dtype=np.int64)

    def instruction_embeddings(self, text):
        if not text:
            raise ValueError("Empty scoring span")
        return np.ones((len(text), 2))

    def _record(self, release, kwargs):
        assert isinstance(release, np.ndarray) and release.shape == (25, 256)
        assert set(kwargs) == {"prompt_text", "scoring_text", "seeds"}
        assert kwargs["seeds"]
        self.calls.append((release.copy(), dict(kwargs)))
        return (.5, .5) if len(self.calls) == 1 else (.1, .1)

    def predict_grounding(self, release, **kwargs):
        coordinate = self._record(release, kwargs)
        candidates = tuple(GroundingCandidate(i, coordinate, True, -.2, .75)
                           for i in range(len(kwargs["seeds"])))
        return GroundingResult(0, coordinate, .5, tuple((c.index, .1) for c in candidates)), candidates

    def predict_action(self, release, *, schemas, **kwargs):
        assert "click" in schemas
        coordinate = self._record(release, kwargs)
        action = {"function": "click", "arguments": {"point": list(coordinate)}, "status": "ok"}
        candidates = tuple(ActionCandidate(i, "click", action["arguments"], "ok", True, -.2, .75)
                           for i in range(len(kwargs["seeds"])))
        return action, .5, candidates


def arguments(**overrides):
    parser = argparse.ArgumentParser()
    add_executor_arguments(parser)
    args = parser.parse_args([])
    args.task = "G"; args.seed = 123; args.family_id = "family-1"; args.device = "cpu"
    args.disable_retrieval = True
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def metadata(name):
    return {"source_task_id": name + "-task", "template_id": name + "-template", "document_id": name + "-doc",
            "normalized_instruction": name + " instruction", "template_family": name + "-tf", "document_family": name + "-df"}


def write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def fixtures(tmp_path, task="G"):
    feature_path = tmp_path / "feature.npy"
    np.save(feature_path, np.zeros((25, 256)), allow_pickle=False)
    rows = []
    for i in range(2):
        row = {"trajectory_id": "recorded-episode", "slot": i, "task": task, "eligible": True,
               "features_path": feature_path.name, "public_metadata": metadata(f"eval{i}")}
        if task == "G":
            row.update(instruction="Locate the requested target", target_box=[.4, .4, .6, .6])
        else:
            row.update(request="Public request", history=[f"thought-{n}" for n in range(12)],
                       reference_action={"function": "click", "arguments": {"point": [.5, .5]}, "status": "ok"},
                       reference_boxes={"point": [.4, .4, .6, .6]}, screen_size=[1000, 1000])
        rows.append(row)
    return write_manifest(tmp_path / "rows.jsonl", rows), rows


def action_schema(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"schema_id": "fixture-v1", "source_split": "fit-train", "source_manifest_sha256": "a" * 64,
        "functions": {"click": {"required": ["point"], "optional": [], "spatial": ["point"], "statuses": ["ok"]}}}), encoding="utf-8")
    return path


def bank_artifacts(tmp_path, population, rows):
    key_path = tmp_path / "bank_keys.npy"
    key = np.zeros((1, 256)); key[0, 0] = 1
    np.save(key_path, key, allow_pickle=False)
    manifest = {"task": "G", "source_split": "fit-train", "source_manifest_sha256": "a" * 64,
                "evaluation_manifest_sha256": file_hash(population), "keys_sha256": file_hash(key_path),
                "success_binding": {"source_field": "fixture_success", "granularity": "trajectory", "missing_policy": "error"},
                "evaluation_public_records": [row["public_metadata"] for row in rows],
                "entries": [{"trajectory_id": "training-episode", "original_slot": 0, "task": "G", "source_split": "fit-train",
                             "eligible": True, "schema_valid": True, "trajectory_success": True,
                             "public_metadata": metadata("train"),
                             "value": {"instruction": "train instruction", "coordinate": [.5, .5], "element_type": "button"}}]}
    manifest_path = tmp_path / "bank.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, key_path


@pytest.mark.parametrize("task", ["G", "A"])
def test_recorded_driver_released_boundary_scores_and_replay_metadata(tmp_path, task):
    manifest, rows = fixtures(tmp_path, task)
    model = FakeModel()
    args = arguments(task=task, action_schema=str(action_schema(tmp_path)) if task == "A" else None)
    score_calls = []
    def scorer(action, slot):
        score_calls.append(slot)
        assert slot.reference_action is not action and slot.screen_size == (1000, 1000)
        return ActionScores(action["function"] == slot.reference_action["function"],
                            action["arguments"] == slot.reference_action["arguments"],
                            action["status"] == slot.reference_action["status"])
    evaluator = RecordedEvaluator(args, model=model, action_evaluator=scorer if task == "A" else None)
    report, episodes = evaluator.run(manifest, output=tmp_path / "result", replicate="2")
    assert report["eligible"] == report["invoked"] == 2
    assert report["correct"] == 1 and report["accuracy"] == .5
    assert report["replicate_id"] == "2" and not report["reproduces_historical_results"]
    assert len(model.calls) == 2 and len(episodes[0]["records"]) == 56
    assert all("reference_action" not in kwargs and "target_box" not in kwargs for _, kwargs in model.calls)
    replay = ReplayBuffer.from_npz(tmp_path / "result/replay.npz")
    assert len(replay.success_indices) == len(replay.failure_indices) == 1
    assert replay.metadata["slot_id"].tolist() == [slot_identifier("recorded-episode", i) for i in range(2)]
    assert replay.metadata["next_slot_id"].tolist() == [slot_identifier("recorded-episode", 1), ""]
    assert replay.metadata["source_manifest_sha256"].tolist() == [file_hash(manifest)] * 2
    assert replay.metadata["task"].tolist() == [task] * 2
    assert replay.metadata["family_id"].tolist() == ["family-1"] * 2
    exports = [json.loads(line) for line in (tmp_path / "result/candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(exports) == 2
    assert len(list((tmp_path / "result/releases").glob("*.npy"))) == 2  # once per invocation, not once per candidate
    assert all(len(row["public_decoder_seeds"]) == len(row["candidates"]) for row in exports)
    if task == "A":
        assert report["function_accuracy"] == report["status_accuracy"] == 1.
        assert report["arguments_accuracy"] == .5 and len(score_calls) == 2
        for _, kwargs in model.calls:
            assert kwargs["scoring_text"].splitlines() == ["Public request"] + rows[0]["history"][-10:]
            assert "thought-0\n" not in kwargs["prompt_text"]
        assert all(row["removed_history"] == 2 for row in exports)


def test_invalid_action_reference_is_rejected_before_private_file_access(tmp_path):
    manifest, rows = fixtures(tmp_path, "A")
    rows[0]["reference_action"]["function"] = "not-in-schema"
    rows[0]["features_path"] = "must-not-be-read.npy"
    write_manifest(manifest, rows)
    model = FakeModel()
    evaluator = RecordedEvaluator(arguments(task="A", action_schema=str(action_schema(tmp_path))),
                                  model=model, action_evaluator=lambda *args: pytest.fail("scored too early"))
    with pytest.raises(ValueError, match="canonical"):
        evaluator.run(manifest)
    assert not model.calls


def test_mandatory_prompt_overflow_precedes_private_screen_access(tmp_path):
    manifest, rows = fixtures(tmp_path)
    rows[0]["instruction"] = "x" * 5000
    rows[0]["features_path"] = "must-not-be-read.npy"
    write_manifest(manifest, rows)
    model = FakeModel()
    with pytest.raises(ValueError, match="cap"):
        RecordedEvaluator(arguments(), model=model).run(manifest)
    assert not model.calls


def test_retrieval_requires_bank_or_explicit_disable():
    with pytest.raises(ValueError, match="explicitly"):
        RecordedEvaluator(arguments(disable_retrieval=False), model=FakeModel())


def test_combined_retrieval_exclusions_require_actual_ids_metadata_and_digest(tmp_path):
    manifest, rows = fixtures(tmp_path)
    combined_rows = rows + [{**rows[0], "trajectory_id": "development-episode", "public_metadata": metadata("development")}]
    combined = write_manifest(tmp_path / "combined.jsonl", combined_rows)
    bank_path, keys_path = bank_artifacts(tmp_path, combined, combined_rows)
    args = arguments(disable_retrieval=False, retrieval_bank=str(bank_path), retrieval_keys=str(keys_path),
                     retrieval_threshold=.25, retrieval_exclusion_manifest=str(combined))
    evaluator = RecordedEvaluator(args, model=FakeModel())
    report, _ = evaluator.run(manifest)
    assert report["retrieval_enabled"] and report["eligible"] == 2
    rows[0]["trajectory_id"] = "unbound-episode"
    rows[0]["features_path"] = "must-not-be-read.npy"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="complete frozen"):
        evaluator.run(manifest)
    combined.write_text(combined.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        evaluator.run(manifest)
