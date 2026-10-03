"""Trusted fit-train inputs for new Stage 1/2 runs, never inference payloads.

Stage 1 follows the declared whole-screen 140x140 native-vision target contract;
there is no crop pooling or guessed alignment. Stage 2 reuses the reference
evaluation prompt renderer, without retrieval, and keeps labels outside prompts.
Preparation does not fit a model or reproduce historical experiment results.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np
from PIL import Image

from .dataset_preparation import _json
from .prompt_policy import prepare_prompt


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def _bound_file(base, binding):
    if (not isinstance(binding, dict) or not isinstance(binding.get("path"), str)
            or not binding["path"] or Path(binding["path"]).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", "")))):
        raise ValueError("Prepared file requires a relative path and SHA256 binding")
    path = (base / binding["path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("Prepared file is missing or outside its artifact directory")
    if _hash(path) != binding["sha256"]:
        raise ValueError(f"Prepared file SHA256 mismatch: {binding['path']}")
    return path


def verify_prepared_inputs(records_path, stage):
    """Validate a preparation sidecar before fitting; return its hash binding.

    Legacy inputs without preparation.json return None. A present sidecar is
    mandatory, never silently ignored on errors. Original source files may have
    moved: this checks prepared payloads against the recorded source digest,
    rather than authenticating a historical execution or re-reading source data.
    """
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError("Prepared-input stage must be 1 or 2")
    records_path = Path(records_path).resolve()
    base = records_path.parent
    sidecar = base / "preparation.json"
    if not sidecar.exists():
        return None
    if not sidecar.is_file():
        raise ValueError("preparation.json must be a file")
    content = sidecar.read_bytes()
    report = _json(content.decode("utf-8"))
    if (not isinstance(report, dict) or type(report.get("format_version")) is not int or report["format_version"] != 1
            or report.get("artifact_kind") != "new-reference-training-inputs"
            or type(report.get("stage")) is not int or report["stage"] != stage):
        raise ValueError("Preparation sidecar kind/version/stage mismatch")
    counts = [report.get(key) for key in ("input_records", "eligible_records", "ineligible_records")]
    if (any(type(n) is not int or n < 0 for n in counts) or counts[1] == 0
            or counts[0] != counts[1] + counts[2]):
        raise ValueError("Preparation record counts are invalid")
    source = report.get("source_manifest")
    if not isinstance(source, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))):
        raise ValueError("Preparation requires source manifest SHA256")
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != counts[1]:
        raise ValueError("Preparation record count differs from eligible_records")
    ids = [row.get("record_id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Preparation record IDs must be nonempty and unique")
    binding = report.get("alignment" if stage == 1 else "records_manifest")
    if _bound_file(base, binding) != records_path:
        raise ValueError("Requested fitting records do not match the preparation sidecar")
    if stage == 1:
        # Read headers, not two potentially multi-GB tensors a second time.
        try:
            with zipfile.ZipFile(records_path) as archive:
                shapes = {}
                for key in ("features", "native_target_tokens", "record_ids"):
                    name = key + ".npy"
                    if archive.namelist().count(name) != 1:
                        raise ValueError("Prepared alignment archive has missing/duplicate arrays")
                    with archive.open(name) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version not in ((1, 0), (2, 0)):
                            raise ValueError("Unsupported prepared NPY format")
                        header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0)
                        shape, _, dtype = header(stream)
                        if dtype.hasobject or (key != "record_ids" and dtype.kind not in "fiu"):
                            raise ValueError("Prepared alignment arrays must not contain objects")
                        shapes[key] = shape
            targets = shapes["native_target_tokens"]
            if (shapes["features"] != (counts[1], 25, 256) or len(targets) != 3
                    or targets[:2] != (counts[1], 25) or targets[2] < 1
                    or shapes["record_ids"] != (counts[1],)):
                raise ValueError("Prepared alignment shapes/counts do not match the sidecar")
            with np.load(records_path, allow_pickle=False) as archive:
                if archive["record_ids"].tolist() != ids:
                    raise ValueError("Prepared alignment record IDs do not match the sidecar")
        except (OSError, EOFError, zipfile.BadZipFile, KeyError) as error:
            raise ValueError("Invalid prepared alignment archive") from error
    else:
        actual = [_json(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(actual) != counts[1] or any(not isinstance(row, dict) for row in actual):
            raise ValueError("Prepared supervised record count/type mismatch")
        if [row.get("record_id") for row in actual] != ids:
            raise ValueError("Prepared supervised record IDs do not match the sidecar")
        eos = report.get("eos_token_id")
        if type(eos) is not int or eos < 0:
            raise ValueError("Preparation sidecar requires its declared EOS")
        for row, metadata in zip(actual, rows):
            for name, count_key in (("prompt_token_ids", "input_tokens"), ("target_token_ids", "target_tokens")):
                tokens = row.get(name)
                if (not isinstance(tokens, list) or not tokens or any(type(i) is not int or i < 0 for i in tokens)
                        or type(metadata.get(count_key)) is not int or len(tokens) != metadata[count_key]):
                    raise ValueError("Prepared token counts/types do not match the sidecar")
            if row["target_token_ids"][-1] != eos or eos in row["target_token_ids"][:-1]:
                raise ValueError("Prepared targets must end in exactly one declared EOS")
            feature = _bound_file(base, {"path": row.get("features_path"), "sha256": metadata.get("features_sha256")})
            value = _array(np.load(feature, allow_pickle=False), (25, 256), "prepared features")
            if np.any(np.linalg.norm(value.astype(np.float64), axis=1) > 1 + 1e-6):
                raise ValueError("Prepared features must be unit-clipped")
    if _hash(sidecar) != hashlib.sha256(content).hexdigest() or _hash(records_path) != binding["sha256"]:
        raise ValueError("Prepared metadata or records changed during verification")
    return {"preparation_path": str(sidecar), "preparation_sha256": hashlib.sha256(content).hexdigest(),
            "stage": stage, "records_sha256": binding["sha256"],
            "source_manifest_sha256": source["sha256"], "eligible_records": counts[1]}


def _training_rows(manifest_path, output_dir, provenance, task=None):
    source, output = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Explicit training-input provenance is required")
    _json(json.dumps(provenance, ensure_ascii=False, allow_nan=False))
    content = source.read_bytes()
    rows, seen, input_count = [], set(), 0
    for line in content.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = _json(line)
        input_count += 1
        if not isinstance(row, dict) or row.get("split") not in ("train", "fit-train"):
            raise ValueError("Every input row must explicitly belong to train/fit-train; dev/test are forbidden")
        if type(row.get("eligible")) is not bool:
            raise ValueError("Every input row requires boolean eligible")
        if row.get("task") not in ("G", "A") or (task is not None and row["task"] != task):
            raise ValueError("Training task must be explicit and match the requested task")
        trajectory, slot = row.get("trajectory_id"), row.get("slot")
        if not isinstance(trajectory, str) or not trajectory.strip() or type(slot) is not int or not 0 <= slot < 56:
            raise ValueError("Training rows require trajectory_id and a slot in [0,55]")
        identifier = row.get("record_id", json.dumps([row["task"], trajectory, slot], ensure_ascii=False))
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("record_id must be nonempty text")
        key = (row["task"], trajectory, slot)
        if identifier in seen or key in seen:
            raise ValueError("Duplicate training record ID or task/trajectory/slot")
        seen.update((identifier, key))
        if row["eligible"]:
            rows.append({**row, "record_id": identifier})
    if not rows:
        raise ValueError("Training manifest contains no eligible fit-train records")
    report = {"format_version": 1, "artifact_kind": "new-reference-training-inputs",
              "reproduces_historical_results": False,
              "scope": "trusted fit-train only; clean features and targets are not released inference inputs",
              "source_manifest": {"path": str(source), "sha256": hashlib.sha256(content).hexdigest()},
              "provenance": provenance, "input_records": input_count,
              "eligible_records": len(rows), "ineligible_records": input_count - len(rows)}
    return source, output, rows, report


@contextmanager
def _publication(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prepare-training-") as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        yield stage
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)


def _path(row, field, base):
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must name an existing local file")
    path = (base / value).resolve()
    if not path.is_file():
        raise ValueError(f"Missing {field}: {path}")
    return path


def _array(value, shape, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite numeric array with shape {shape}")
    return array


def _features(row, base, encoder, *, allow_native_image=False):
    fields = [key for key in ("image_path", "features_path") if key in row]
    if allow_native_image and len(fields) == 2:
        fields = ["features_path"]
    if len(fields) != 1:
        raise ValueError("Training rows require exactly one image_path or features_path")
    field = fields[0]
    path = _path(row, field, base)
    digest = _hash(path)
    if field == "features_path":
        value = np.load(path, allow_pickle=False)
    elif encoder is not None:
        value = encoder(path)
    else:
        raise ValueError("Image inputs require an explicitly supplied DINO feature encoder")
    features = _array(value, (25, 256), "clean features").astype(np.float64)
    if np.any(np.linalg.norm(features, axis=1) > 1 + 1e-6):
        raise ValueError("Clean training features must already be unit-clipped")
    if _hash(path) != digest:
        raise ValueError("Training feature/image source changed during preparation")
    return features, {"path": str(path), "sha256": digest, "field": field}


class QwenNativeTargetEncoder:
    """Frozen whole-screen native vision, used only for alignment supervision."""

    def __init__(self, vision, image_processor, *, device="cpu", provenance):
        self.vision = vision.to(device).eval().requires_grad_(False)
        self.image_processor = image_processor
        self.device = device
        self.provenance = dict(provenance)
        if getattr(vision, "spatial_merge_size", None) != 2:
            raise ValueError("Native targets require Qwen spatial_merge_size=2")

    def __call__(self, image_path):
        import torch
        with Image.open(image_path) as image:
            if min(image.size) < 5:
                raise ValueError("Screen must have at least five pixels per dimension")
            screen = image.convert("RGB").resize((140, 140), Image.Resampling.BICUBIC)
        batch = self.image_processor(images=[screen], do_resize=False, return_tensors="pt")
        grid = batch["image_grid_thw"]
        if grid.shape != (1, 3) or grid.tolist() != [[1, 10, 10]]:
            raise ValueError("Native processor must preserve exactly image_grid_thw=[[1,10,10]]")
        dtype = next(self.vision.parameters()).dtype
        with torch.inference_mode():
            tokens = self.vision(batch["pixel_values"].to(device=self.device, dtype=dtype),
                                 grid_thw=grid.to(self.device))
        if tokens.ndim != 2 or tokens.shape[0] != 25 or not torch.isfinite(tokens).all():
            raise ValueError("Native Qwen vision must return 25 finite row-major post-merge tokens")
        return tokens.detach().cpu().float().numpy()

    @classmethod
    def from_pretrained(cls, model_id, revision=None, *, device="cpu", dtype="float32",
                        local_files_only=True):
        import torch
        from transformers import AutoImageProcessor
        from .executor import _immutable_model_ref, released_qwen_class
        _immutable_model_ref(model_id, revision)
        if dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("Unsupported native-vision dtype")
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, use_fast=False,
            local_files_only=local_files_only, trust_remote_code=False)
        model = released_qwen_class().from_pretrained(model_id, revision=revision,
            local_files_only=local_files_only, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), attn_implementation="eager")
        # Retain only the frozen vision module; language weights are not needed here.
        return cls(model.model.visual, processor, device=device, provenance={
            "model": str(model_id), "revision": revision or "local-explicit-checkpoint",
            "dtype": dtype, "processor": processor.to_dict(),
            "target_transform": "whole RGB screen bicubic 140x140; native post-merge row-major 25 tokens"})


def build_alignment_data(manifest_path, output_dir, *, feature_encoder=None, native_encoder=None,
                         provenance):
    """Publish alignment.npz, containing features[N,25,256] and targets[N,25,H].

    Each eligible row supplies native_target_tokens_path, or image_path plus a
    native_encoder. features_path may coexist with image_path for native targets;
    it takes precedence for clean DINO features. Supplied targets must already follow the declared row order.
    Large target arrays are disk-backed while constructing the NPZ.
    """
    source, output, rows, report = _training_rows(manifest_path, output_dir, provenance)
    report.update(stage=1, native_encoder=getattr(native_encoder, "provenance", None),
                  target_alignment="25 row-major post-merge positions; supplied targets are caller-bound",
                  records=[])
    with _publication(output) as stage:
        # Temporary memmaps avoid stacking every screen's 25x3584 targets in RAM.
        with tempfile.TemporaryDirectory(dir=stage.parent, prefix="arrays-") as temporary:
            features_map = targets_map = None
            try:
                for index, row in enumerate(rows):
                    features, feature_binding = _features(row, source.parent, feature_encoder, allow_native_image=True)
                    if "native_target_tokens_path" in row:
                        path = _path(row, "native_target_tokens_path", source.parent)
                        digest = _hash(path)
                        target = np.load(path, allow_pickle=False)
                        target_kind = "supplied-row-major-native-tokens"
                    else:
                        if native_encoder is None or "image_path" not in row:
                            raise ValueError("Stage 1 needs native_target_tokens_path or image_path plus native_encoder")
                        path = _path(row, "image_path", source.parent)
                        digest = _hash(path)
                        if feature_binding["field"] == "image_path" and digest != feature_binding["sha256"]:
                            raise ValueError("Screen changed between feature and native-target extraction")
                        target = native_encoder(path)
                        target_kind = "whole-screen-native-vision"
                    if hasattr(target, "detach"):
                        target = target.detach().cpu().float().numpy()
                    target = np.asarray(target)
                    if target.ndim != 2 or target.shape[1] < 1:
                        raise ValueError("Native targets require shape (25, hidden_size)")
                    target = _array(target, (25, target.shape[1]), "native targets").astype(np.float32)
                    if not np.isfinite(target).all() or _hash(path) != digest:
                        raise ValueError("Native target is nonfinite or its source changed")
                    if features_map is None:
                        features_map = np.lib.format.open_memmap(Path(temporary) / "features.npy", mode="w+",
                            dtype=np.float64, shape=(len(rows), 25, 256))
                        targets_map = np.lib.format.open_memmap(Path(temporary) / "targets.npy", mode="w+",
                            dtype=np.float32, shape=(len(rows), 25, target.shape[1]))
                    if target.shape != targets_map.shape[1:]:
                        raise ValueError("All native targets must have the same hidden size")
                    features_map[index], targets_map[index] = features, target
                    report["records"].append({"record_id": row["record_id"], "features": feature_binding,
                        "native_target": {"path": str(path), "sha256": digest, "kind": target_kind}})
                np.savez_compressed(stage / "alignment.npz", features=features_map,
                                    native_target_tokens=targets_map,
                                    record_ids=np.asarray([r["record_id"] for r in rows]))
            finally:
                for array in (features_map, targets_map):
                    if array is not None:
                        array._mmap.close()
        report["alignment"] = {"path": "alignment.npz", "sha256": _hash(stage / "alignment.npz")}
        if _hash(source) != report["source_manifest"]["sha256"]:
            raise ValueError("Source training manifest changed during preparation")
        _write_json(stage / "preparation.json", report)
    return report


def load_training_tokenizer(model_id, revision=None, *, local_files_only=True):
    from transformers import AutoTokenizer
    from .executor import _immutable_model_ref
    _immutable_model_ref(model_id, revision, kind="tokenizer")
    return AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_files_only,
                                        trust_remote_code=False)


def _target_text(row, task, schemas):
    from .scoring import parse_action
    text = row.get("target_text")
    if text is None and task == "A" and "reference_action" in row:
        text = json.dumps(row["reference_action"], ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Stage 2 requires explicit target_text; Grounding boxes are never converted to center targets")
    value = _json(text)
    if task == "G":
        if (not isinstance(value, dict) or set(value) != {"x", "y"}
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1
                       for v in value.values())):
            raise ValueError("Grounding target_text must be canonical normalized {x,y} JSON")
    else:
        parsed = parse_action(text, schemas)
        if parsed is None or parsed != value or ("reference_action" in row and row["reference_action"] != value):
            raise ValueError("Action target must already be canonical under the supplied frozen schema")
    return text


def _visual_ids(tokenizer):
    names = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>")
    ids = tuple(tokenizer.convert_tokens_to_ids(name) for name in names)
    if any(type(i) is not int or i < 0 or i == getattr(tokenizer, "unk_token_id", None) for i in ids) or len(set(ids)) != 4:
        raise ValueError("Tokenizer must define distinct native vision token IDs")
    return ids


def _prompt_ids(tokenizer, payload, schema_text, visual_ids):
    from .evaluation import render_prompt
    marker = "<|vision_start|>" + "<|image_pad|>" * 25 + "<|vision_end|>"
    messages = [{"role": "user", "content": marker + "\n" + render_prompt(payload, action_schema_text=schema_text)}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Chat template must return actual integer token IDs")
    start, image, end, video = visual_ids
    positions = [i for i, token in enumerate(ids) if token == image]
    if (len(positions) != 25 or ids.count(start) != 1 or ids.count(end) != 1 or video in ids
            or positions != list(range(positions[0], positions[0] + 25))
            or positions[0] == 0 or positions[-1] + 1 >= len(ids)
            or ids[positions[0] - 1] != start or ids[positions[-1] + 1] != end):
        raise ValueError("Prompt must preserve one contiguous 25-pad native visual wrapper")
    return ids


def build_supervised_data(manifest_path, output_dir, *, task, tokenizer, provenance,
                          eos_token_id=None, feature_encoder=None, action_schema=None):
    """Publish records.jsonl plus clean NPY features for existing fit-adapter.

    Prompt construction has no label input. Grounding requires target_text;
    Action requires a supplied schema and target_text or reference_action.
    EOS defaults to the tokenizer's single EOS; ambiguous EOS needs an explicit ID.
    """
    if task not in ("G", "A"):
        raise ValueError("task must be G or A")
    source, output, rows, report = _training_rows(manifest_path, output_dir, provenance, task)
    from .action_evaluation import load_action_schemas
    if (task == "A") != (action_schema is not None):
        raise ValueError("Action requires a frozen action_schema; Grounding must not supply one")
    schema_bytes = Path(action_schema).read_bytes() if action_schema is not None else None
    schemas = load_action_schemas(action_schema) if action_schema is not None else None
    # Match RecordedEvaluator exactly, including the caller's schema serialization.
    schema_text = schema_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n") if schema_bytes else ""
    declared = getattr(tokenizer, "eos_token_id", None)
    declared = [declared] if type(declared) is int else declared
    if (not isinstance(declared, (list, tuple, set, frozenset)) or not declared
            or any(type(i) is not int or i < 0 for i in declared)):
        raise ValueError("Tokenizer must declare valid EOS token IDs; an unknown EOS cannot be overridden")
    allowed_eos = set(declared)
    eos = next(iter(allowed_eos)) if eos_token_id is None and len(allowed_eos) == 1 else eos_token_id
    if type(eos) is not int or eos not in allowed_eos:
        raise ValueError("Select a single EOS token ID from the tokenizer's declared EOS set")
    visual = _visual_ids(tokenizer)
    if allowed_eos.intersection(visual):
        raise ValueError("Declared EOS must not be a visual token")
    special = set(tokenizer.all_special_ids) | set(visual) | allowed_eos
    prepared = []
    # Preflight all text and labels before private image loading or output creation.
    for row in rows:
        target = _target_text(row, task, schemas)
        current = row.get("instruction") if task == "G" else row.get("request")
        history = row.get("history", [])
        if not isinstance(history, list) or (task == "A" and "history" not in row):
            raise ValueError("Action requires explicit chronological history (possibly [])")
        prompt = prepare_prompt(task, current, history, (),
            lambda payload: _prompt_ids(tokenizer, payload, schema_text, visual))
        scoring = tokenizer.encode(prompt.payload.scoring_text, add_special_tokens=False)
        targets = tokenizer.encode(target, add_special_tokens=False)
        if any(not ids or any(type(i) is not int or i < 0 or i in special for i in ids) for ids in (scoring, targets)):
            raise ValueError("Instruction and complete target must have nonempty nonspecial token spans")
        prepared.append((prompt, targets + [eos]))
    report.update(stage=2, task=task, eos_token_id=eos,
                  action_schema_sha256=hashlib.sha256(schema_bytes).hexdigest() if schema_bytes else None,
                  prompt_policy="jdc-reference-v1; evaluation render_prompt; retrieval disabled for supervised fitting",
                  target_policy="explicit normalized Grounding JSON; canonical schema-bound Action; append one EOS",
                  records=[])
    with _publication(output) as stage:
        (stage / "features").mkdir()
        with (stage / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for index, (row, (prompt, targets)) in enumerate(zip(rows, prepared)):
                features, binding = _features(row, source.parent, feature_encoder)
                path = f"features/{index:08d}.npy"
                np.save(stage / path, features, allow_pickle=False)
                record = {"record_id": row["record_id"], "features_path": path,
                          "prompt_token_ids": list(prompt.input_ids), "target_token_ids": targets}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                report["records"].append({"record_id": row["record_id"], "features": binding,
                    "features_sha256": _hash(stage / path), "retained_history_positions": list(prompt.retained_history_positions),
                    "input_tokens": len(prompt.input_ids), "target_tokens": len(targets)})
        report["records_manifest"] = {"path": "records.jsonl", "sha256": _hash(stage / "records.jsonl")}
        if _hash(source) != report["source_manifest"]["sha256"]:
            raise ValueError("Source training manifest changed during preparation")
        _write_json(stage / "preparation.json", report)
    return report
