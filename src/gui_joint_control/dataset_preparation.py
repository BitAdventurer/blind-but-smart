"""Prepare supplied normalized records with explicit trajectory split assignments.

This validates a small local interchange format, not an upstream benchmark
converter or a reconstruction of the manuscript's unavailable split assignment.
No private image/feature bytes are read, and no experiments are run.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


ROLES = ("train", "dev", "test")


def _json(text):
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=reject)
    # Also reject numeric overflow (1e999) and escaped invalid Unicode before output.
    json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return value


def _identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("trajectory_id must be a nonempty string")
    return value


def _split_ids(payload):
    if not isinstance(payload, dict) or set(payload) != set(ROLES):
        raise ValueError("Split assignment requires exactly train, dev and test ID arrays")
    assignments = {}
    for role in ROLES:
        if not isinstance(payload[role], list):
            raise ValueError(f"{role} split must be a trajectory ID array")
        for raw in payload[role]:
            identifier = _identifier(raw)
            if identifier in assignments:
                raise ValueError(f"Duplicate or overlapping split trajectory: {identifier}")
            assignments[identifier] = role
    return assignments


def _screen_size(value):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value)):
        raise ValueError("screen_size must be positive integer [width, height]")
    return value


def _box(value, limits=(1, 1)):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or (isinstance(v, float) and not math.isfinite(v)) for v in value)
            or not 0 <= value[0] <= value[2] <= limits[0]
            or not 0 <= value[1] <= value[3] <= limits[1]):
        raise ValueError("Target box must be finite ordered [x_min,y_min,x_max,y_max] within the screen")
    return [value[0] / limits[0], value[1] / limits[1],
            value[2] / limits[0], value[3] / limits[1]]


def _record(row, *, task, role, source_dir, output_dir, schemas):
    value = dict(row)
    if value.get("task", task) != task:
        raise ValueError("Record task differs from requested dataset task")
    if "split" in value and value["split"] != role:
        raise ValueError("Record split conflicts with the explicit trajectory assignment")
    if "eligible" not in value or type(value["eligible"]) is not bool:
        raise ValueError("Every record requires explicit boolean eligible")
    eligible = value["eligible"]
    value.update(task=task, split=role)
    if "screen_size" in value:
        _screen_size(value["screen_size"])

    paths = [field for field in ("image_path", "features_path") if field in value]
    if len(paths) > 1 or (eligible and len(paths) != 1):
        raise ValueError("Eligible records require exactly one image_path or features_path")
    for field in paths + (["native_target_tokens_path"] if "native_target_tokens_path" in value else []):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{field} must be a nonempty local file path")
        path = (source_dir / value[field]).resolve()
        if eligible and not path.is_file():
            raise ValueError(f"Missing eligible {field}: {path}")
        value[field] = Path(os.path.relpath(path, output_dir)).as_posix()

    if task == "G":
        if "thought" in value:
            if "instruction" in value:
                raise ValueError("Supply either instruction or thought, not both")
            value["instruction"] = value.pop("thought")
        if ("instruction" in value and not isinstance(value["instruction"], str)) or (
                eligible and not value.get("instruction", "").strip()):
            raise ValueError("Eligible Grounding records require nonempty instruction or thought")
        if "target_box" in value and "target_box_pixels" in value:
            raise ValueError("Supply either normalized target_box or target_box_pixels, not both")
        if "target_box_pixels" in value:
            if "screen_size" not in value:
                raise ValueError("target_box_pixels requires explicit screen_size")
            value["target_box"] = _box(value.pop("target_box_pixels"), value["screen_size"])
        elif value.get("target_box") is not None:
            value["target_box"] = _box(value["target_box"])
        elif eligible:
            raise ValueError("Eligible Grounding records require an offline target box")
    else:
        if ("request" in value and not isinstance(value["request"], str)) or (
                eligible and not value.get("request", "").strip()):
            raise ValueError("Eligible Action records require explicit nonempty request")
        if "history" not in value and eligible:
            raise ValueError("Eligible Action records require explicit chronological history (possibly [])")
        history = value.get("history", [])
        if not isinstance(history, list) or any(not isinstance(v, str) or not v.strip() for v in history):
            raise ValueError("history must contain chronological nonempty recorded thought strings")
        reference = value.get("reference_action")
        if reference is not None:
            if (not isinstance(reference, dict) or set(reference) != {"function", "arguments", "status"}
                    or any(not isinstance(reference[k], str) or not reference[k].strip()
                           or reference[k] == "INVALID" for k in ("function", "status"))
                    or not isinstance(reference["arguments"], dict)):
                raise ValueError("reference_action must already be a canonical function/arguments/status object")
            if schemas is not None:
                from .scoring import parse_action
                if parse_action(json.dumps(reference, allow_nan=False), schemas) != reference:
                    raise ValueError("reference_action is not canonical under the supplied frozen schema")
        elif eligible:
            raise ValueError("Eligible Action records require an offline reference_action")
        if "reference_boxes" in value and not isinstance(value["reference_boxes"], dict):
            raise ValueError("reference_boxes must be an explicitly supplied offline mapping")
    return value


def prepare_dataset(records_path, splits_path, output_dir, task="G", *, action_schema=None):
    """Atomically write train/dev/test manifests and source-bound preparation counts.

    Validation errors refuse the entire request before creating output. Only
    trajectories exceeding the fixed 56-slot horizon are exclusions; they are
    excluded whole and recorded in exclusions.jsonl. Existing output is refused.
    """
    if task not in ("G", "A"):
        raise ValueError("Expected Grounding (G) or Action (A)")
    records_path, splits_path, output_dir = map(lambda p: Path(p).resolve(),
                                                (records_path, splits_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if action_schema is not None and task != "A":
        raise ValueError("action_schema only applies to Action")
    records_bytes, splits_bytes = records_path.read_bytes(), splits_path.read_bytes()
    assignments = _split_ids(_json(splits_bytes.decode("utf-8-sig")))
    schemas = None
    schema_binding = None
    if action_schema is not None:
        from .action_evaluation import load_action_schemas
        schema_path = Path(action_schema).resolve()
        schemas = load_action_schemas(schema_path)
        schema_binding = {"path": str(schema_path), "sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest()}
    groups = {}
    for number, line in enumerate(records_bytes.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = _json(line)
        if not isinstance(row, dict):
            raise ValueError(f"Record line {number} must be a JSON object")
        trajectory = _identifier(row.get("trajectory_id"))
        slot = row.get("slot")
        if type(slot) is not int or slot < 0:
            raise ValueError(f"Record line {number}: slot must be a zero-based nonnegative integer")
        if trajectory not in assignments:
            raise ValueError(f"Missing explicit split assignment for trajectory: {trajectory}")
        group = groups.setdefault(trajectory, {})
        if slot in group:
            raise ValueError(f"Duplicate trajectory/slot: {trajectory}/{slot}")
        group[slot] = _record(row, task=task, role=assignments[trajectory], source_dir=records_path.parent,
                              output_dir=output_dir, schemas=schemas)
    if not groups:
        raise ValueError("Empty records file")
    if set(assignments) != set(groups):
        raise ValueError("Split assignment includes trajectories absent from the records")

    retained = {role: [] for role in ROLES}
    exclusions = []
    for trajectory, group in sorted(groups.items()):
        role = assignments[trajectory]
        if len(group) > 56 or max(group) >= 56:
            exclusions.append({"trajectory_id": trajectory, "split": role, "reason": "exceeds_56_slot_horizon",
                               "record_count": len(group), "max_slot": max(group)})
        else:
            retained[role].extend(row for _, row in sorted(group.items()))
    counts = {}
    for role, rows in retained.items():
        rejected = [row for row in exclusions if row["split"] == role]
        counts[role] = {"accepted_trajectories": len({r["trajectory_id"] for r in rows}),
                       "accepted_records": len(rows), "eligible_records": sum(r["eligible"] for r in rows),
                       "ineligible_records": sum(not r["eligible"] for r in rows),
                       "excluded_trajectories": len(rejected),
                       "excluded_records": sum(r["record_count"] for r in rejected)}
    encoded = {f"{role}.jsonl": "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in rows) for role, rows in retained.items()}
    report = {"format_version": 1, "task": task, "reproduces_historical_results": False,
              "source": {"path": str(records_path), "sha256": hashlib.sha256(records_bytes).hexdigest()},
              "split_assignment": {"path": str(splits_path), "sha256": hashlib.sha256(splits_bytes).hexdigest(),
                                   "rule": "explicit trajectory IDs; no inferred or random assignment"},
              "action_schema": schema_binding, "input_records": sum(map(len, groups.values())),
              "input_trajectories": len(groups), "counts": counts,
              "manifests": {role: {"path": f"{role}.jsonl", "sha256": hashlib.sha256(encoded[f"{role}.jsonl"].encode("utf-8")).hexdigest()}
                            for role in ROLES},
              "reference_scope": "trusted offline labels; not executor or controller inputs"}
    encoded["exclusions.jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exclusions)
    encoded["preparation.json"] = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".prepare-dataset-") as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        for name, text in encoded.items():
            (stage / name).write_text(text, encoding="utf-8", newline="\n")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        stage.rename(output_dir)
    return report
