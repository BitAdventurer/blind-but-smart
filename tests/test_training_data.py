"""Tiny trusted-input fixtures; no downloaded weights or benchmark measurements."""
import hashlib
import json
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from gui_joint_control.training_data import (
    QwenNativeTargetEncoder, build_alignment_data, build_supervised_data, load_training_tokenizer,
    verify_prepared_inputs,
)


PROVENANCE = {"kind": "tiny-software-fixture", "model_revision": "a" * 40}


class Tokenizer:
    eos_token_id = 63
    pad_token_id = 0
    unk_token_id = 2
    tokens = {"<|vision_start|>": 3, "<|image_pad|>": 4, "<|vision_end|>": 5, "<|video_pad|>": 6}
    all_special_ids = [0, 1, 2, 3, 4, 5, 6, 63]

    def __init__(self):
        self.messages = []

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, 2)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        for part in re.split(r"(<\|[^>]+\|>)", text):
            if part in self.tokens:
                ids.append(self.tokens[part])
            elif part == "<|eos|>":
                ids.append(63)
            else:
                ids.extend(100 + ord(c) for c in part)
        return ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt and len(messages) == 1
        assert set(messages[0]) == {"role", "content"} and messages[0]["role"] == "user"
        self.messages.append(messages[0]["content"])
        return [1] + self.encode(messages[0]["content"]) + [7]


def manifest(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def record(tmp_path, **updates):
    np.save(tmp_path / "features.npy", np.eye(25, 256) * .5, allow_pickle=False)
    return {"task": "G", "split": "train", "trajectory_id": "train-1", "slot": 0,
            "eligible": True, "features_path": "features.npy", "instruction": "Click Save",
            "target_text": '{"x":0.125,"y":0.75}', **updates}


def schema(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"schema_id": "tiny", "source_split": "fit-train",
        "source_manifest_sha256": "b" * 64, "functions": {"click": {
            "required": ["point"], "optional": [], "spatial": ["point"], "statuses": ["ok"]}}},
        indent=2), encoding="utf-8")
    return path


def test_alignment_supplied_arrays_consumed_by_existing_trainer(tmp_path):
    from gui_joint_control.fitting import AlignmentExample, Stage1Trainer, initialize_projection
    targets = np.eye(25, 32, dtype=np.float32)
    np.save(tmp_path / "native.npy", targets, allow_pickle=False)
    row = record(tmp_path, native_target_tokens_path="native.npy", record_id="r-1")
    source = manifest(tmp_path, [row, {**row, "slot": 1, "record_id": "skip", "eligible": False}])
    output = tmp_path / "alignment"
    report = build_alignment_data(source, output, provenance=PROVENANCE)
    with np.load(output / "alignment.npz", allow_pickle=False) as data:
        assert data["features"].shape == (1, 25, 256)
        np.testing.assert_array_equal(data["native_target_tokens"][0], targets)
        assert data["record_ids"].tolist() == ["r-1"]
        examples = [AlignmentExample("r-1", data["features"][0], data["native_target_tokens"][0])]
        history = Stage1Trainer(initialize_projection(32, seed=3), seed=4).fit(examples, epochs=1)
    assert len(history) == 1 and np.isfinite(history[0]["loss"])
    assert report["input_records"] == 2 and report["eligible_records"] == 1
    assert report["source_manifest"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["records"][0]["native_target"]["sha256"] == hashlib.sha256((tmp_path / "native.npy").read_bytes()).hexdigest()
    before = (output / "preparation.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_alignment_data(source, output, provenance=PROVENANCE)
    assert (output / "preparation.json").read_bytes() == before


def test_whole_screen_native_encoder_and_dino_callback_alignment(tmp_path):
    import torch

    class Vision(torch.nn.Module):
        spatial_merge_size = 2

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(25 * 32).reshape(25, 32).float())

        def forward(self, pixels, *, grid_thw):
            assert not torch.is_grad_enabled() and grid_thw.tolist() == [[1, 10, 10]]
            assert pixels.shape == (100, 1176)
            return self.weight

    pixels = np.zeros((30, 50, 3), dtype=np.uint8)
    pixels[:, :25, 0] = 255
    pixels[:, 25:, 2] = 255
    Image.fromarray(pixels).save(tmp_path / "screen.png")

    def processor(*, images, do_resize, return_tensors):
        assert len(images) == 1 and images[0].size == (140, 140) and images[0].mode == "RGB"
        assert images[0].getpixel((0, 0)) == (255, 0, 0)
        assert images[0].getpixel((139, 0)) == (0, 0, 255)
        assert do_resize is False and return_tensors == "pt"
        return {"pixel_values": torch.zeros(100, 1176), "image_grid_thw": torch.tensor([[1, 10, 10]])}

    calls = []
    def features(path):
        calls.append(path.name)
        return np.eye(25, 256)

    native = QwenNativeTargetEncoder(Vision(), processor, provenance=PROVENANCE)
    assert not native.vision.training and not native.vision.weight.requires_grad
    row = record(tmp_path)
    row.pop("features_path")
    row["image_path"] = "screen.png"
    output = tmp_path / "out"
    report = build_alignment_data(manifest(tmp_path, [row]), output,
                                  feature_encoder=features, native_encoder=native, provenance=PROVENANCE)
    with np.load(output / "alignment.npz") as archive:
        np.testing.assert_array_equal(archive["native_target_tokens"][0], native.vision.weight.numpy())
    assert calls == ["screen.png"]
    assert report["records"][0]["native_target"]["kind"] == "whole-screen-native-vision"


def test_native_encoder_actual_pinned_processor_and_tiny_vision(tmp_path):
    import torch
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    config = Qwen2_5_VLVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
        num_heads=4, out_hidden_size=32, spatial_merge_size=2, fullatt_block_indexes=[0])
    config._attn_implementation = "eager"
    vision = Qwen2_5_VisionTransformerPretrainedModel(config)
    processor = Qwen2VLImageProcessor()
    Image.new("RGB", (40, 60), (10, 20, 30)).save(tmp_path / "screen.png")
    tokens = QwenNativeTargetEncoder(vision, processor, provenance=PROVENANCE)(tmp_path / "screen.png")
    assert tokens.shape == (25, 32) and np.isfinite(tokens).all()
    assert not any(p.requires_grad for p in vision.parameters())


def test_supervised_matches_executor_prompt_and_masks_labels(tmp_path):
    import torch
    from gui_joint_control.executor import ReleasedQwenExecutor
    from gui_joint_control.evaluation import render_prompt
    from gui_joint_control.prompt_policy import PromptPayload
    from gui_joint_control.fitting import teacher_forced_inputs, initialize_projection

    tokenizer = Tokenizer()
    row = record(tmp_path)
    output = tmp_path / "out"
    report = build_supervised_data(manifest(tmp_path, [row]), output, task="G", tokenizer=tokenizer,
                                   provenance=PROVENANCE)
    saved = json.loads((output / "records.jsonl").read_text())
    config = SimpleNamespace(image_token_id=4, video_token_id=6, vision_start_token_id=3, vision_end_token_id=5)
    shim = SimpleNamespace(tokenizer=tokenizer, device="cpu", model=SimpleNamespace(config=config))
    expected = ReleasedQwenExecutor.prompt_token_ids(shim, render_prompt(PromptPayload("G", row["instruction"], (), ())))
    assert saved["prompt_token_ids"] == expected[0].tolist()
    assert saved["target_token_ids"] == tokenizer.encode(row["target_text"]) + [63]
    assert all(row["target_text"] not in message for message in tokenizer.messages)
    embedding = torch.nn.Embedding(512, 32)
    model = SimpleNamespace(config=config, get_input_embeddings=lambda: embedding,
        generation_config=SimpleNamespace(eos_token_id=63, pad_token_id=0),
        model=SimpleNamespace(get_rope_index=lambda ids, **kw:
            (torch.arange(ids.shape[1]).expand(3, 1, -1), torch.zeros(1, 1))))
    clean = np.load(output / saved["features_path"], allow_pickle=False)
    inputs = teacher_forced_inputs(model, initialize_projection(32, seed=7), clean,
                                  saved["prompt_token_ids"], saved["target_token_ids"])
    n_prompt = len(saved["prompt_token_ids"])
    assert (inputs["labels"][0, :n_prompt] == -100).all()
    assert inputs["labels"][0, n_prompt:].tolist() == saved["target_token_ids"]
    assert report["eos_token_id"] == 63 and report["stage"] == 2


def test_action_history_and_exact_schema_are_public_prompt_only(tmp_path):
    tokenizer = Tokenizer()
    reference = {"function": "click", "arguments": {"point": [.23, .41]}, "status": "ok"}
    row = record(tmp_path, task="A", request="Click next", history=[f"step-{i}" for i in range(12)],
                 reference_action=reference)
    row.pop("target_text")
    path = schema(tmp_path)
    out = tmp_path / "out"
    report = build_supervised_data(manifest(tmp_path, [row]), out, task="A", tokenizer=tokenizer,
                                   provenance=PROVENANCE, action_schema=path)
    prompt = tokenizer.messages[-1]
    assert path.read_text() in prompt
    assert "step-0\n" not in prompt and "step-1\n" not in prompt
    assert "\n".join(row["history"][-10:]) in prompt
    assert '"point": [0.23, 0.41]' not in prompt
    saved = json.loads((out / "records.jsonl").read_text())
    canonical = json.dumps(reference, separators=(",", ":"))
    assert saved["target_token_ids"] == tokenizer.encode(canonical) + [63]
    assert report["records"][0]["retained_history_positions"] == list(range(2, 12))


@pytest.mark.parametrize("split", [None, "dev", "test", "evaluation"])
def test_nontraining_rows_rejected_even_if_ineligible_before_encoder(tmp_path, split):
    good = record(tmp_path)
    bad = {**good, "slot": 1, "eligible": False, "split": split}
    if split is None:
        bad.pop("split")
    def forbidden(*args):
        pytest.fail("Encoder must not run before split validation")
    with pytest.raises(ValueError, match="train/fit-train"):
        build_alignment_data(manifest(tmp_path, [good, bad]), tmp_path / "out",
                             feature_encoder=forbidden, native_encoder=forbidden, provenance=PROVENANCE)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change,match", [
    ({"target_text": None}, "explicit target_text"),
    ({"target_text": '{"x":2,"y":0}'}, "normalized"),
    ({"target_text": '{"x":NaN,"y":0}'}, "Nonfinite"),
    ({"target_text": '{"x":0,"x":1,"y":0}'}, "Duplicate"),
    ({"instruction": "<|image_pad|>"}, "visual wrapper"),
    ({"instruction": "x" * 5000}, "exceeds the cap"),
    ({"instruction": "<|eos|>"}, "nonspecial"),
])
def test_text_preflight_rejects_entire_input_before_feature_access(tmp_path, change, match):
    good = record(tmp_path)
    bad = {**good, "slot": 1, **change}
    for row in (good, bad):
        row.pop("features_path")
        row["image_path"] = "never-open.png"
    with pytest.raises(ValueError, match=match):
        build_supervised_data(manifest(tmp_path, [good, bad]), tmp_path / "out", task="G",
            tokenizer=Tokenizer(), provenance=PROVENANCE, feature_encoder=lambda path: pytest.fail("No image access"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("target", ["null", '{"function":"other","arguments":{},"status":"ok"}'])
def test_action_invalid_target_is_not_accepted_as_none(tmp_path, target):
    row = record(tmp_path, task="A", request="Click", history=[], target_text=target)
    with pytest.raises(ValueError, match="canonical"):
        build_supervised_data(manifest(tmp_path, [row]), tmp_path / "out", task="A",
            tokenizer=Tokenizer(), provenance=PROVENANCE, action_schema=schema(tmp_path))


@pytest.mark.parametrize("kind", ["shape", "nan", "unclipped", "native_shape", "native_hidden"])
def test_alignment_bad_arrays_abort_without_partial_output(tmp_path, kind):
    row = record(tmp_path, native_target_tokens_path="native.npy")
    features = np.eye(25, 256) * .5
    target = np.eye(25, 32)
    if kind == "shape": features = features[:24]
    if kind == "nan": features[0, 0] = np.nan
    if kind == "unclipped": features[0, 0] = 2
    if kind == "native_shape": target = target[:24]
    np.save(tmp_path / "features.npy", features, allow_pickle=False)
    np.save(tmp_path / "native.npy", target, allow_pickle=False)
    rows = [row]
    if kind == "native_hidden":
        np.save(tmp_path / "other.npy", np.zeros((25, 16)), allow_pickle=False)
        rows.append({**row, "slot": 1, "native_target_tokens_path": "other.npy"})
    with pytest.raises(ValueError):
        build_alignment_data(manifest(tmp_path, rows), tmp_path / "out", provenance=PROVENANCE)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".prepare-training-*"))


def test_loader_defaults_are_local_only_and_revision_pinned(monkeypatch):
    import torch
    import transformers
    from gui_joint_control import executor
    calls = []
    class Vision(torch.nn.Linear):
        spatial_merge_size = 2
    vision = Vision(1, 1)
    def processor(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"do_normalize": True})
    def model(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=SimpleNamespace(visual=vision))
    def tokenizer(*args, **kwargs):
        calls.append(kwargs)
        return Tokenizer()
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", processor)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(executor, "released_qwen_class", lambda: SimpleNamespace(from_pretrained=model))
    for loader in (QwenNativeTargetEncoder.from_pretrained, load_training_tokenizer):
        with pytest.raises(ValueError, match="immutable"):
            loader("example/remote", "main")
    assert not calls
    QwenNativeTargetEncoder.from_pretrained("example/remote", "a" * 40)
    load_training_tokenizer("example/remote", "a" * 40)
    assert len(calls) == 3 and all(c["local_files_only"] is True and c["trust_remote_code"] is False for c in calls)
    assert all(c["revision"] == "a" * 40 for c in calls)


def test_supervised_overwrite_and_incomplete_eos_refused(tmp_path):
    source = manifest(tmp_path, [record(tmp_path)])
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        build_supervised_data(source, tmp_path / "out", task="G", tokenizer=tokenizer, provenance=PROVENANCE)
    with pytest.raises(ValueError, match="EOS"):
        build_supervised_data(source, tmp_path / "out", task="G", tokenizer=tokenizer, provenance=PROVENANCE, eos_token_id=63)
    tokenizer.eos_token_id = 63
    build_supervised_data(source, tmp_path / "out", task="G", tokenizer=tokenizer, provenance=PROVENANCE, eos_token_id=63)
    before = (tmp_path / "out" / "records.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_supervised_data(source, tmp_path / "out", task="G", tokenizer=tokenizer, provenance=PROVENANCE, eos_token_id=63)
    assert (tmp_path / "out" / "records.jsonl").read_bytes() == before


@pytest.mark.parametrize("declared,selected", [(63, 42), (63, True), ([63, 64], None),
    ({63, 64}, 42), ([], 63), ([63, True], 63)])
def test_supervised_unknown_or_ambiguous_eos_rejected(tmp_path, declared, selected):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = declared
    with pytest.raises(ValueError, match="EOS"):
        build_supervised_data(manifest(tmp_path, [record(tmp_path)]), tmp_path / "out", task="G",
            tokenizer=tokenizer, eos_token_id=selected, provenance=PROVENANCE)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("declared,selected,expected", [([63], None, 63), ([63, 64], 64, 64),
    ({63, 64}, 63, 63), ((63, 64), 64, 64)])
def test_supervised_declared_eos_choice(tmp_path, declared, selected, expected):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = declared
    output = tmp_path / "out"
    report = build_supervised_data(manifest(tmp_path, [record(tmp_path)]), output, task="G",
        tokenizer=tokenizer, eos_token_id=selected, provenance=PROVENANCE)
    saved = json.loads((output / "records.jsonl").read_text())
    assert saved["target_token_ids"][-1] == expected == report["eos_token_id"]


def test_alignment_precomputed_features_and_separate_native_image(tmp_path):
    Image.new("RGB", (30, 50), (10, 20, 30)).save(tmp_path / "screen.png")
    row = record(tmp_path, image_path="screen.png")
    calls = []
    def native(path):
        calls.append(path.name)
        return np.eye(25, 32)
    output = tmp_path / "out"
    report = build_alignment_data(manifest(tmp_path, [row]), output, native_encoder=native,
        feature_encoder=lambda path: pytest.fail("Precomputed features take precedence"), provenance=PROVENANCE)
    assert calls == ["screen.png"]
    assert report["records"][0]["features"]["field"] == "features_path"
    for field, path in (("features", "features.npy"), ("native_target", "screen.png")):
        assert report["records"][0][field]["sha256"] == hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
    with np.load(output / "alignment.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["features"][0], np.eye(25, 256) * .5)
        np.testing.assert_array_equal(archive["native_target_tokens"][0], np.eye(25, 32))


@pytest.mark.parametrize("stage", [1, 2])
def test_source_manifest_change_during_preparation_aborts(tmp_path, stage):
    row = record(tmp_path, image_path="screen.png")
    row.pop("features_path")
    Image.new("RGB", (30, 50)).save(tmp_path / "screen.png")
    source = manifest(tmp_path, [row])
    def features(path):
        source.write_text(source.read_text() + "\n", encoding="utf-8")
        return np.eye(25, 256)
    with pytest.raises(ValueError, match="Source training manifest changed"):
        if stage == 1:
            build_alignment_data(source, tmp_path / "out", feature_encoder=features,
                native_encoder=lambda path: np.eye(25, 32), provenance=PROVENANCE)
        else:
            build_supervised_data(source, tmp_path / "out", task="G", tokenizer=Tokenizer(),
                feature_encoder=features, provenance=PROVENANCE)
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".prepare-training-*"))


def prepared_fixture(tmp_path, stage):
    row = record(tmp_path, record_id="r-1")
    output = tmp_path / "out"
    if stage == 1:
        np.save(tmp_path / "native.npy", np.eye(25, 32), allow_pickle=False)
        row["native_target_tokens_path"] = "native.npy"
        build_alignment_data(manifest(tmp_path, [row]), output, provenance=PROVENANCE)
        return output / "alignment.npz"
    build_supervised_data(manifest(tmp_path, [row]), output, task="G", tokenizer=Tokenizer(), provenance=PROVENANCE)
    return output / "records.jsonl"


@pytest.mark.parametrize("stage", [1, 2])
def test_verify_prepared_inputs_binds_saved_payload_and_legacy_none(tmp_path, stage):
    assert verify_prepared_inputs(tmp_path / "legacy", stage) is None
    path = prepared_fixture(tmp_path, stage)
    sidecar = path.parent / "preparation.json"
    report = json.loads(sidecar.read_text())
    binding = verify_prepared_inputs(path, stage)
    assert binding == {"preparation_path": str(sidecar),
        "preparation_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(), "stage": stage,
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_manifest_sha256": report["source_manifest"]["sha256"], "eligible_records": 1}


@pytest.mark.parametrize("stage", [1, 2])
def test_verify_prepared_inputs_payload_hash_mismatch(tmp_path, stage):
    path = prepared_fixture(tmp_path, stage)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_prepared_inputs(path, stage)


@pytest.mark.parametrize("change,match", [
    ({"stage": 1}, "kind/version/stage"),
    ({"artifact_kind": "historical-data"}, "kind/version/stage"),
    ({"format_version": True}, "kind/version/stage"),
    ({"eligible_records": 2}, "counts"),
    ({"records": []}, "count"),
    ({"records": [{"record_id": "other"}]}, "IDs"),
    ({"source_manifest": {"sha256": "invalid"}}, "source manifest"),
])
def test_verify_preparation_metadata_mismatch(tmp_path, change, match):
    path = prepared_fixture(tmp_path, 2)
    sidecar = path.parent / "preparation.json"
    report = json.loads(sidecar.read_text())
    report.update(change)
    sidecar.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        verify_prepared_inputs(path, 2)


@pytest.mark.parametrize("change,match", [("feature_hash", "SHA256 mismatch"),
    ("feature_shape", "shape"), ("record_id", "IDs"), ("token_count", "token counts"),
    ("eos", "one declared EOS"), ("outside_path", "outside"), ("duplicate_id", "unique")])
def test_verify_supervised_structure_even_when_payload_hash_updated(tmp_path, change, match):
    path = prepared_fixture(tmp_path, 2)
    sidecar = path.parent / "preparation.json"
    report = json.loads(sidecar.read_text())
    saved = json.loads(path.read_text())
    features = path.parent / saved["features_path"]
    if change == "feature_hash":
        np.save(features, np.zeros((25, 256)), allow_pickle=False)
    elif change == "feature_shape":
        np.save(features, np.zeros((24, 256)), allow_pickle=False)
        report["records"][0]["features_sha256"] = hashlib.sha256(features.read_bytes()).hexdigest()
    elif change == "record_id":
        saved["record_id"] = "other"
    elif change == "token_count":
        saved["prompt_token_ids"].append(101)
    elif change == "eos":
        saved["target_token_ids"][-1] = 42
    elif change == "outside_path":
        saved["features_path"] = "../features.npy"
    else:
        report["records"].append(report["records"][0])
        report.update(input_records=2, eligible_records=2)
    path.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    report["records_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        verify_prepared_inputs(path, 2)


@pytest.mark.parametrize("change,match", [("shape", "shapes/counts"), ("id", "IDs"), ("object", "objects")])
def test_verify_alignment_structure_even_when_archive_hash_updated(tmp_path, change, match):
    path = prepared_fixture(tmp_path, 1)
    sidecar = path.parent / "preparation.json"
    report = json.loads(sidecar.read_text())
    features = np.zeros((1, 25, 256))
    targets = np.zeros((1, 25, 32))
    ids = np.asarray(["r-1"])
    if change == "shape": targets = targets[:, :24]
    if change == "id": ids = np.asarray(["other"])
    if change == "object": features = features.astype(object)
    np.savez_compressed(path, features=features, native_target_tokens=targets, record_ids=ids)
    report["alignment"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        verify_prepared_inputs(path, 1)
