"""Train-only bank admission, global views, and protected-query contracts."""
import hashlib
import json
import numpy as np
import pytest

from gui_joint_control.retrieval import FrozenDemonstrationBank, canonical_json, serialize_demonstrations


def public_metadata(prefix):
    return {"source_task_id": prefix + "-task", "template_id": prefix + "-template", "document_id": prefix + "-doc",
            "template_family": prefix + "-template-family", "document_family": prefix + "-doc-family",
            "normalized_instruction": prefix + " different"}


def artifacts(tmp_path, *, count=10):
    keys = np.zeros((count, 256)); keys[:, 0] = 2.
    key_path = tmp_path / "keys.npy"; np.save(key_path, keys, allow_pickle=False)
    rows = []
    for i in range(count):
        rows.append({"trajectory_id": f"t{i:03}", "original_slot": 1, "task": "G", "source_split": "fit-train",
                     "eligible": True, "schema_valid": True, "trajectory_success": True,
                     "public_metadata": public_metadata(f"train{i}"),
                     "value": {"instruction": f"click {i}", "coordinate": [.5, .5], "element_type": "button"}})
    manifest = {"task": "G", "source_split": "fit-train", "source_manifest_sha256": "a" * 64,
                "evaluation_manifest_sha256": "b" * 64,
                "keys_sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
                "success_binding": {"source_field": "fixture.success", "granularity": "trajectory", "missing_policy": "error"},
                "evaluation_public_records": [public_metadata("eval")], "entries": rows}
    manifest_path = tmp_path / "bank.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, key_path, manifest


def test_release_query_gate_top_eight_order_and_zero_mean(tmp_path):
    manifest_path, key_path, _ = artifacts(tmp_path)
    bank = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    release = np.zeros((25, 256)); release[:, 0] = 1
    assert not bank.retrieve(release, .25, .25)  # strict greater-than gate
    entries = bank.retrieve(release, .5, .25)
    assert len(entries) == 8
    assert [json.loads(e.text)["instruction"] for e in entries] == [f"click {i}" for i in range(8)]
    assert all(e.score == 1. for e in entries)
    assert not bank.retrieve(np.zeros((25, 256)), 2., 0.)
    with pytest.raises(ValueError, match="Threshold"):
        bank.retrieve(release, .5, .3)
    with pytest.raises(ValueError):
        bank._keys[0, 0] = 0.
    with pytest.raises(ValueError):
        bank._keys.setflags(write=True)
    encoded = serialize_demonstrations(entries)
    assert len(json.loads(encoded)) == 8 and "trajectory_success" not in encoded and "public_metadata" not in encoded


def test_global_exact_and_strict_views_are_frozen_before_query(tmp_path):
    manifest_path, key_path, manifest = artifacts(tmp_path, count=4)
    manifest["entries"][0]["public_metadata"]["document_id"] = "eval-doc"
    manifest["entries"][1]["public_metadata"]["template_family"] = "eval-template-family"
    manifest["entries"][2]["public_metadata"]["normalized_instruction"] = "one two three four five"
    manifest["evaluation_public_records"][0]["normalized_instruction"] = "one two three four five six"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    primary = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G", view="primary")
    strict = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G", view="strict")
    assert [e.trajectory_id for e in primary.entries] == ["t001", "t002", "t003"]
    assert [e.trajectory_id for e in strict.entries] == ["t003"]
    assert strict.provenance["evaluation_manifest_sha256"] == "b" * 64


def test_train_only_success_missing_policy_and_digest(tmp_path):
    manifest_path, key_path, manifest = artifacts(tmp_path, count=3)
    manifest["entries"][0]["trajectory_success"] = False
    manifest["entries"][1]["trajectory_success"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="success labels"):
        FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    manifest["success_binding"]["missing_policy"] = "exclude"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bank = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    assert len(bank.entries) == 1
    manifest["entries"][2]["source_split"] = "test"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fit-train"):
        FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    key_path.write_bytes(key_path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="digest"):
        FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")


def test_canonical_nfc_serialization_does_not_normalize_current_request():
    assert canonical_json({"z": "e\u0301", "a": .15}) == '{"a":0.15,"z":"é"}'
    with pytest.raises(ValueError, match="duplicate"):
        canonical_json({"é": 1, "e\u0301": 2})
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})


def test_bank_binds_complete_exclusion_manifest_and_rejects_metadata_omissions(tmp_path):
    manifest_path, key_path, manifest = artifacts(tmp_path, count=2)
    population = tmp_path / "evaluation.jsonl"
    rows = [{"trajectory_id": "eval-trajectory", "slot": 0, "task": "G",
             "public_metadata": public_metadata("eval")}]
    population.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest["evaluation_manifest_sha256"] = hashlib.sha256(population.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bank = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    assert bank.validate_evaluation_manifest(population) == manifest["evaluation_manifest_sha256"]
    manifest["evaluation_public_records"] = [public_metadata("unrelated")]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="G")
    with pytest.raises(ValueError, match="metadata"):
        wrong.validate_evaluation_manifest(population)
    population.write_text(population.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        bank.validate_evaluation_manifest(population)


def test_action_bank_requires_schema_and_canonical_values(tmp_path):
    from gui_joint_control.scoring import ActionFunctionSchema
    manifest_path, key_path, manifest = artifacts(tmp_path, count=1)
    manifest["task"] = "A"
    manifest["entries"][0]["task"] = "A"
    manifest["entries"][0]["value"] = {"instruction": "request", "schema_id": "fixture-v1",
        "reference_action": {"function": "click", "arguments": {"point": [.5, .5]}, "status": "ok"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="schemas"):
        FrozenDemonstrationBank.from_files(manifest_path, key_path, task="A")
    schemas = {"click": ActionFunctionSchema(frozenset({"point"}), frozenset(), frozenset({"point"}), frozenset({"ok"}))}
    bank = FrozenDemonstrationBank.from_files(manifest_path, key_path, task="A", schemas=schemas)
    assert bank.entries[0].task == "A"
    manifest["entries"][0]["value"]["reference_action"]["function"] = "unknown"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        FrozenDemonstrationBank.from_files(manifest_path, key_path, task="A", schemas=schemas)
