"""Local interchange validation; fixture records are not benchmark results."""
import hashlib
import json

import pytest

from gui_joint_control.dataset_preparation import prepare_dataset
from gui_joint_control.evaluation import manifest_rows


def inputs(tmp_path, rows=None, splits=None):
    source = tmp_path / "supplied"
    source.mkdir()
    (source / "screen.png").write_bytes(b"fixture path only; no private asset is decoded")
    records = source / "records.jsonl"
    if rows is None:
        rows = [record("training"), record("development"), record("testing")]
    records.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assignment = source / "splits.json"
    assignment.write_text(json.dumps(splits if splits is not None else {
        "train": ["training"], "dev": ["development"], "test": ["testing"]}), encoding="utf-8")
    return records, assignment, tmp_path / "prepared"


def record(trajectory="training", slot=0, **values):
    return {"trajectory_id": trajectory, "slot": slot, "eligible": True, "instruction": "Find Save",
            "target_box": [.1, .2, .3, .4], "image_path": "screen.png", **values}


def test_grounding_normalizes_explicit_pixel_box_preserves_slots_and_source(tmp_path):
    pixel = record("training", slot=2, screen_size=[200, 100], target_box_pixels=[20, 20, 60, 40])
    del pixel["target_box"]
    pixel["thought"] = pixel.pop("instruction")
    pixel["public_metadata"] = {"source_task_id": "fixed-source", "nested": [1, 2]}
    skip = {"trajectory_id": "training", "slot": 1, "eligible": False, "reason": "supplied exclusion"}
    rows = [pixel, skip, record("development"), record("testing")]
    records, assignment, output = inputs(tmp_path, rows)
    original_records, original_assignment = records.read_bytes(), assignment.read_bytes()
    report = prepare_dataset(records, assignment, output)
    train = manifest_rows(output / "train.jsonl")["training"]
    assert set(train) == {1, 2} and not train[1]["eligible"]
    assert train[1]["reason"] == "supplied exclusion"
    assert train[2]["instruction"] == "Find Save" and "thought" not in train[2]
    assert train[2]["target_box"] == pytest.approx([.1, .2, .3, .4])
    assert "target_box_pixels" not in train[2]
    assert train[2]["public_metadata"] == pixel["public_metadata"]
    assert (output / train[2]["image_path"]).resolve() == records.parent / "screen.png"
    assert train[2]["image_path"].startswith("../")
    assert report["source"]["sha256"] == hashlib.sha256(original_records).hexdigest()
    assert report["split_assignment"]["sha256"] == hashlib.sha256(original_assignment).hexdigest()
    assert report["counts"]["train"]["accepted_records"] == 2
    assert report["counts"]["train"]["eligible_records"] == 1
    assert report["counts"]["train"]["ineligible_records"] == 1
    assert records.read_bytes() == original_records and assignment.read_bytes() == original_assignment
    for role, trajectory in [("train", "training"), ("dev", "development"), ("test", "testing")]:
        assert set(manifest_rows(output / f"{role}.jsonl")) == {trajectory}
        assert report["manifests"][role]["sha256"] == hashlib.sha256((output / f"{role}.jsonl").read_bytes()).hexdigest()
    assert not report["reproduces_historical_results"]


@pytest.mark.parametrize("long_rows", [57, 2])
def test_excludes_whole_trajectory_over_56_slots_with_report(tmp_path, long_rows):
    long = [record("long", slot=i) for i in range(57)] if long_rows == 57 else [record("long"), record("long", slot=56)]
    records, assignment, output = inputs(tmp_path, long + [record("training")],
        {"train": ["training", "long"], "dev": [], "test": []})
    report = prepare_dataset(records, assignment, output)
    assert set(manifest_rows(output / "train.jsonl")) == {"training"}
    excluded = json.loads((output / "exclusions.jsonl").read_text(encoding="utf-8"))
    assert excluded == {"trajectory_id": "long", "split": "train", "reason": "exceeds_56_slot_horizon",
                        "record_count": long_rows, "max_slot": 56}
    assert report["input_records"] == long_rows + 1
    assert report["counts"]["train"]["excluded_records"] == long_rows
    assert report["counts"]["train"]["excluded_trajectories"] == 1


@pytest.mark.parametrize("splits", [
    {"train": ["training"], "dev": ["training"], "test": []},
    {"train": ["training", "training"], "dev": [], "test": []},
    {"train": ["training"], "test": []},
    {"train": [], "dev": [], "test": []},
    {"train": ["training", "absent"], "dev": [], "test": []},
])
def test_refuses_leaking_missing_or_unknown_split_ids_before_output(tmp_path, splits):
    records, assignment, output = inputs(tmp_path, [record()], splits)
    with pytest.raises(ValueError):
        prepare_dataset(records, assignment, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepare-dataset-*"))


@pytest.mark.parametrize("change", [
    {"eligible": None}, {"slot": True}, {"slot": -1}, {"instruction": ""},
    {"target_box": [.3, .2, .1, .4]}, {"target_box": [-.1, .2, .3, .4]},
    {"target_box": [False, .2, .3, .4]}, {"target_box_pixels": [1, 2, 3, 4]},
    {"thought": "ambiguous second public text"}, {"features_path": "also-supplied.npy"},
    {"image_path": "missing.png"}, {"split": "test"}, {"task": "A"},
])
def test_refuses_ambiguous_or_invalid_records_before_output(tmp_path, change):
    records, assignment, output = inputs(tmp_path, [record(**change)], {"train": ["training"], "dev": [], "test": []})
    with pytest.raises(ValueError):
        prepare_dataset(records, assignment, output)
    assert not output.exists()


def test_duplicate_slots_and_output_collision_preserve_inputs(tmp_path):
    records, assignment, output = inputs(tmp_path, [record(), record()], {"train": ["training"], "dev": [], "test": []})
    with pytest.raises(ValueError, match="Duplicate trajectory/slot"):
        prepare_dataset(records, assignment, output)
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_dataset(records, assignment, output)
    assert sentinel.read_text(encoding="utf-8") == "existing data"


def test_action_requires_canonical_reference_and_preserves_supplied_history(tmp_path):
    row = {"trajectory_id": "training", "slot": 0, "task": "A", "eligible": True,
           "features_path": "features.npy", "request": "Save the document",
           "history": ["Opened the document", "Found the Save menu"],
           "reference_action": {"function": "save", "arguments": {"filename": "draft"}, "status": "ok"},
           "screen_size": [100, 200], "reference_boxes": {"supplied": [1, 2, 3, 4]},
           "public_metadata": {"template_id": "unchanged"}}
    records, assignment, output = inputs(tmp_path, [row], {"train": ["training"], "dev": [], "test": []})
    (records.parent / "features.npy").write_bytes(b"fixture feature path")
    report = prepare_dataset(records, assignment, output, task="A")
    prepared = manifest_rows(output / "train.jsonl", "A")["training"][0]
    assert prepared["history"] == row["history"] and prepared["request"] == row["request"]
    assert prepared["reference_action"] == row["reference_action"]
    assert prepared["reference_boxes"] == row["reference_boxes"]
    assert prepared["screen_size"] == row["screen_size"]
    assert prepared["public_metadata"] == row["public_metadata"]
    assert (output / prepared["features_path"]).resolve() == records.parent / "features.npy"
    assert report["action_schema"] is None


@pytest.mark.parametrize("reference,history", [
    ({"function": "save", "arguments": {}, "status": "ok", "extra": "ambiguous"}, []),
    ({"function": "save", "arguments": {}, "status": "ok"}, None),
    ({"function": "save", "arguments": {}, "status": "ok"}, ["valid thought", ""]),
])
def test_action_rejects_noncanonical_or_ambiguous_inputs(tmp_path, reference, history):
    row = {"trajectory_id": "training", "slot": 0, "eligible": True, "image_path": "screen.png",
           "request": "Save", "reference_action": reference}
    if history is not None:
        row["history"] = history
    records, assignment, output = inputs(tmp_path, [row], {"train": ["training"], "dev": [], "test": []})
    with pytest.raises(ValueError):
        prepare_dataset(records, assignment, output, task="A")
    assert not output.exists()


def test_action_optional_schema_rejects_alias_repair(tmp_path):
    row = {"trajectory_id": "training", "slot": 0, "eligible": True, "image_path": "screen.png",
           "request": "Save", "history": [],
           "reference_action": {"function": "alias", "arguments": {}, "status": "ok"}}
    records, assignment, output = inputs(tmp_path, [row], {"train": ["training"], "dev": [], "test": []})
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"schema_id": "supplied-v1", "source_split": "fit-train", "source_manifest_sha256": "a" * 64,
        "functions": {"save": {"required": [], "optional": [], "spatial": [], "statuses": ["ok"]}},
        "function_aliases": {"alias": "save"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        prepare_dataset(records, assignment, output, task="A", action_schema=schema)
    assert not output.exists()
    row["reference_action"]["function"] = "save"
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = prepare_dataset(records, assignment, output, task="A", action_schema=schema)
    assert report["action_schema"]["sha256"] == hashlib.sha256(schema.read_bytes()).hexdigest()
    assert manifest_rows(output / "train.jsonl", "A")["training"][0]["reference_action"] == row["reference_action"]


@pytest.mark.parametrize("line", [
    '{"trajectory_id":"training","trajectory_id":"other","slot":0,"eligible":false}',
    '{"trajectory_id":"training","slot":0,"eligible":false,"metadata":NaN}',
    '{"trajectory_id":"training","slot":0,"eligible":false,"metadata":1e999}',
])
def test_rejects_duplicate_json_keys_and_nonfinite_metadata_before_output(tmp_path, line):
    records, assignment, output = inputs(tmp_path, [record()], {"train": ["training"], "dev": [], "test": []})
    records.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_dataset(records, assignment, output)
    assert not output.exists()
