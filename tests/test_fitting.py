"""Tiny random CPU fitting checks; none measure paper benchmark performance."""
import json
import numpy as np
import pytest

from gui_joint_control.fitting import (
    AlignmentExample, TeacherForcedExample, Stage1Trainer, Stage2Trainer,
    initialize_projection, symmetric_alignment_loss, teacher_forced_inputs,
)
from gui_joint_control.executor import ReleasedQwenExecutor, released_qwen_class


def test_alignment_uses_position_negatives_and_stops_target_gradient():
    import torch
    correct = torch.eye(25, requires_grad=True)
    target = torch.eye(25, requires_grad=True)
    matched = symmetric_alignment_loss(correct, target)
    mismatched = symmetric_alignment_loss(correct, target.roll(1, 0))
    assert matched < mismatched
    matched.backward()
    assert correct.grad is not None
    assert target.grad is None


def test_stage1_two_epochs_updates_projection_and_writes_safe_artifacts(tmp_path):
    import torch
    projection = initialize_projection(32, seed=1)
    before = projection.weight.detach().clone()
    rng = np.random.default_rng(4)
    records = [AlignmentExample("train-1", np.eye(25, 256), rng.normal(size=(25, 32))),
               AlignmentExample("train-2", np.eye(25, 256) * .5, rng.normal(size=(25, 32)))]
    trainer = Stage1Trainer(projection, seed=8)
    history = trainer.fit(records)
    assert len(history) == 4
    assert all(2 <= h["mean_refinement_epsilon"] <= 4 for h in history)
    assert not torch.equal(before, projection.weight)
    trainer.save(tmp_path, provenance={"kind": "tiny-random-test"})
    assert np.load(tmp_path / "projection.npy", allow_pickle=False).shape == (32, 256)
    assert json.loads((tmp_path / "fit_metadata.json").read_text())["artifact_kind"] == "new-reference-fit"
    with pytest.raises(FileExistsError):
        trainer.save(tmp_path, provenance={})


def tiny_model():
    import torch
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig)
    torch.manual_seed(177)
    config = Qwen2_5_VLConfig(
        text_config=Qwen2_5_VLTextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            rope_scaling={"type": "default", "mrope_section": [1, 1, 2]}).to_dict(),
        vision_config=Qwen2_5_VLVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                            num_heads=4, out_hidden_size=32,
                                            spatial_merge_size=2, fullatt_block_indexes=[0]).to_dict(),
        image_token_id=4, video_token_id=6, vision_start_token_id=3, vision_end_token_id=5)
    config._attn_implementation = "eager"
    model = released_qwen_class()(config)
    model.generation_config.eos_token_id = 63
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 1
    return model


def test_stage2_real_qwen_complete_target_loss_and_gradients(tmp_path):
    import torch
    import torch.nn.functional as F
    base = tiny_model()
    initial_base = {name: p.detach().clone() for name, p in base.named_parameters()}
    projection = initialize_projection(32, seed=99)
    before_projection = projection.weight.detach().clone()
    trainer = Stage2Trainer(base, projection, seed=55)
    trained = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
    assert trained and all(".language_model." in name and "lora_" in name for name in trained)
    assert all(("q_proj" in name or "v_proj" in name) for name in trained)
    prompt = [1, 3] + [4] * 25 + [5, 12, 13]
    target = [10, 11, 63]
    clean = np.eye(25, 256)
    inputs = teacher_forced_inputs(trainer.model, projection, clean, prompt, target)
    assert int((inputs["labels"] != -100).sum()) == len(target)
    assert (inputs["labels"][0, :len(prompt)] == -100).all()
    def forbid_vision(*args, **kwargs):
        raise AssertionError("Stage 2 must not run the native vision tower")
    base.model.visual.forward = forbid_vision
    output = trainer.model(**inputs)
    expected = F.cross_entropy(output.logits[0, len(prompt) - 1:-1].float(), torch.tensor(target))
    assert output.loss.detach().item() == pytest.approx(expected.detach().item(), rel=1e-6)
    history = trainer.fit([TeacherForcedExample("train-1", clean, prompt, target)])
    assert len(history) == 3
    assert all(.5 <= h["mean_refinement_epsilon"] <= 4 for h in history)
    assert not torch.equal(before_projection, projection.weight)
    assert any(not torch.equal(trained[name], p) for name, p in trainer.model.named_parameters() if name in trained)
    # Original base matrices (including wrapped q/v base_layer) are unchanged.
    for name, p in trainer.model.get_base_model().named_parameters():
        canonical = name.replace(".base_layer.", ".")
        if canonical in initial_base:
            assert torch.equal(p, initial_base[canonical]), name
    trainer.save(tmp_path, provenance={"kind": "tiny-random-fit"})
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    trainer.model.eval()
    with torch.no_grad():
        before_merge = trainer.model(**teacher_forced_inputs(trainer.model, projection, clean, prompt, target)).logits
    merged, weights = trainer.merge_for_inference()
    assert weights.shape == (32, 256)
    assert not any(p.requires_grad for p in merged.parameters())
    with torch.no_grad():
        after_merge = merged(**teacher_forced_inputs(merged, projection, clean, prompt, target)).logits
    torch.testing.assert_close(before_merge, after_merge, rtol=1e-5, atol=1e-6)


def test_teacher_forcing_rejects_incomplete_targets():
    base = tiny_model()
    projection = initialize_projection(32, seed=3)
    prompt = [1, 3] + [4] * 25 + [5, 12]
    with pytest.raises(ValueError, match="EOS"):
        teacher_forced_inputs(base, projection, np.zeros((25, 256)), prompt, [10])
    with pytest.raises(ValueError, match="unpadded"):
        teacher_forced_inputs(base, projection, np.zeros((25, 256)), prompt, [10, 0, 63])


def test_bfloat16_qwen_keeps_fp32_projection_gradients_and_runs_frozen_inference():
    import torch
    base = tiny_model().to(dtype=torch.bfloat16)
    projection = initialize_projection(32, seed=188)
    before = projection.weight.detach().clone()
    trainer = Stage2Trainer(base, projection, seed=441)
    prompt = [1, 3] + [4] * 25 + [5, 12, 13]
    target = [10, 11, 63]
    clean = np.eye(25, 256)
    inputs = teacher_forced_inputs(trainer.model, projection, clean, prompt, target)
    assert inputs["inputs_embeds"].dtype == torch.bfloat16
    history = trainer.fit([TeacherForcedExample("bf16-software-test", clean, prompt, target)], epochs=1)
    assert len(history) == 1 and np.isfinite(history[0]["loss"])
    assert projection.weight.dtype == torch.float32
    assert projection.weight.grad.dtype == torch.float32
    assert torch.isfinite(projection.weight.grad).all() and projection.weight.grad.abs().sum() > 0
    assert not torch.equal(before, projection.weight)
    merged, weights = trainer.merge_for_inference()
    assert merged.get_input_embeddings().weight.dtype == torch.bfloat16

    class DecodeOnlyTokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))

    executor = ReleasedQwenExecutor(merged, DecodeOnlyTokenizer(), weights,
                                    provenance={"kind": "tiny-bf16-software-test"})
    seen_dtypes = []
    def inspect(_model, args, kwargs):
        seen_dtypes.append(kwargs["inputs_embeds"].dtype)
    hook = merged.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        result = executor.generate_tokens(clean, torch.tensor([prompt]), seeds=[991], max_new_tokens=2)
    finally:
        hook.remove()
    assert len(result) == 1 and np.isfinite(result[0].logprob_mean)
    assert seen_dtypes and all(dtype == torch.bfloat16 for dtype in seen_dtypes)
    assert not any(parameter.requires_grad for parameter in merged.parameters())
