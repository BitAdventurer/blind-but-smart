"""Software verification only: the tiny model is random, not a paper result."""
import math
import numpy as np
import pytest

from gui_joint_control.scoring import (
    ActionCandidate, ActionFunctionSchema, GroundingCandidate, aggregate_action,
    aggregate_grounding, cell_index, parse_action, parse_grounding, score_relevance,
)
from gui_joint_control.executor import ReleasedQwenExecutor, released_qwen_class, _immutable_model_ref


def candidate(i, xy, lp=0.0, r=1.0, valid=True):
    return GroundingCandidate(i, xy, valid, lp, r)


def test_same_cell_signed_relevance_and_zero_cases():
    release = np.zeros((25, 256))
    release[0, 0] = 1.0
    projection = np.zeros((4, 256)); projection[0, 0] = 1.0
    embeddings = np.array([[1., 0., 0., 0.]])
    assert score_relevance(release, projection, embeddings, [(0.01, 0.01)]) == 1.0
    assert score_relevance(release, projection, embeddings, [(0.19, 0.19)]) == 1.0
    assert score_relevance(release, projection, -embeddings, [(0.1, 0.1)]) == -1.0
    assert score_relevance(release * 0, projection, embeddings, [(0.1, 0.1)]) == 0.5
    assert score_relevance(release * 0, projection, embeddings * 0, [(0.1, 0.1)]) == 1.0
    assert cell_index((1.0, 1.0)) == 24
    with pytest.raises(ValueError):
        score_relevance(release, projection, np.empty((0, 4)), [])


def test_relevance_spatial_keys_are_repeated_before_projection():
    release = np.zeros((25, 256)); release[0, 0] = 1; release[1, 1] = 1
    projection = np.eye(2, 256)
    actual = score_relevance(release, projection, [[1., 0.]], [(0, 0), (0.2, 0), (0.01, 0)])
    assert actual == pytest.approx(2 / math.sqrt(5))


def test_sampled_medoid_and_missing_support_feedback():
    candidates = [candidate(0, (0., 0.)), candidate(1, (1., 1.), lp=-4),
                  candidate(2, None, valid=False)]
    result = aggregate_grounding(candidates)
    assert result.selected_index == 0
    assert result.coordinate == (0., 0.)
    assert result.feedback == pytest.approx(4 / 3)
    assert aggregate_grounding([candidate(0, None, valid=False)]).feedback == 2.0
    single = aggregate_grounding([candidate(0, (.5, .5)), candidate(1, None, valid=False)])
    assert single.feedback == 1.0


def test_raw_squared_signed_weights_and_absolute_tie():
    # Opposed visual/text vectors have r=-1 and retain full weight by contract.
    result = aggregate_grounding([candidate(5, (1., 0.), r=-1), candidate(3, (0., 0.), r=1)])
    assert result.selected_index == 3
    assert dict(result.weights) == {3: 1.0, 5: 1.0}
    result = aggregate_grounding([candidate(2, (0., 0.), lp=-100, r=0), candidate(1, (1., 0.), lp=-100, r=0)])
    assert result.selected_index == 1
    with pytest.raises(ValueError):
        aggregate_grounding([candidate(0, (0., 0.), lp=1)])


@pytest.mark.parametrize("text", ['{"x":0,"x":1,"y":0}', '{"x":NaN,"y":0}',
                                  '{"x":true,"y":0}', 'prefix {"x":0,"y":0}'])
def test_grounding_parser_rejects_ambiguous_output(text):
    assert parse_grounding(text) is None


def test_grounding_clips_finite_coordinates_but_rejects_truncation():
    assert parse_grounding('{"x":-1,"y":2}') == (0., 1.)
    assert parse_grounding('{"x":0,"y":0}', terminated=False) is None


def test_action_aggregation_remains_sampled_and_missing_support_counts():
    schema = {"click": ActionFunctionSchema(frozenset({"point"}), frozenset(), frozenset({"point"}), frozenset({"ok"}))}
    items = [ActionCandidate(0, "click", {"point": [.1, .2]}, "ok", True, 0., .5),
             ActionCandidate(1, "click", {"point": [.8, .9]}, "ok", True, -3., .5),
             ActionCandidate(2, "INVALID", {}, "INVALID", False, 0., 0.)]
    output, feedback = aggregate_action(items, schema)
    assert output["arguments"]["point"] == [.1, .2]
    assert 0 <= feedback <= 2
    assert parse_action('{"function":"unknown","arguments":{},"status":"ok"}', schema) is None


def test_no_mutable_remote_revision():
    with pytest.raises(ValueError, match="immutable"):
        _immutable_model_ref("Qwen/Qwen2.5-VL-7B-Instruct", "main")
    _immutable_model_ref("Qwen/Qwen2.5-VL-7B-Instruct", "a" * 40)


def test_tokenizer_only_directory_uses_its_own_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    _immutable_model_ref(str(tmp_path), None, kind="tokenizer")
    with pytest.raises(ValueError, match="config.json"):
        _immutable_model_ref(str(tmp_path), None, kind="model")
    with pytest.raises(ValueError, match="immutable"):
        _immutable_model_ref("Qwen/Qwen2.5-VL-7B-Instruct", "main", kind="tokenizer")


class TinyTokenizer:
    """No-download tokenizer fixture; the model/attention/cache are real Qwen."""
    all_special_ids = [0, 1, 2, 3, 4, 5, 6, 63]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["content"].count("<|image_pad|>") == 25
        return [1, 3] + [4] * 25 + [5, 10]

    def encode(self, text, add_special_tokens=False):
        return [10 + ord(c) % 40 for c in text]

    def decode(self, tokens, **kwargs):
        return " ".join(str(t) for t in tokens if t not in self.all_special_ids)


@pytest.fixture
def tiny_executor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig)
    torch.manual_seed(17)
    config = Qwen2_5_VLConfig(
        text_config=Qwen2_5_VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=256, rope_scaling={"type": "default", "mrope_section": [1, 1, 2]}).to_dict(),
        vision_config=Qwen2_5_VLVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                            num_heads=4, out_hidden_size=32, spatial_merge_size=2,
                                            fullatt_block_indexes=[0]).to_dict(),
        image_token_id=4, video_token_id=6, vision_start_token_id=3, vision_end_token_id=5,
    )
    config._attn_implementation = "eager"
    model = released_qwen_class()(config).eval()
    model.generation_config.eos_token_id = 63
    model.generation_config.pad_token_id = 0
    model.generation_config.bos_token_id = 1
    # Uniform raw logits make complete-output probability analytically checkable.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    projection = np.zeros((32, 256)); projection[:32, :32] = np.eye(32)
    return ReleasedQwenExecutor(model, TinyTokenizer(), projection,
                                provenance={"kind": "tiny-random-software-test"})


def test_real_tiny_qwen_prefill_cache_and_no_vision_call(tiny_executor):
    import torch
    executor = tiny_executor
    release = np.ones((25, 256)) * .2
    input_ids = executor.prompt_token_ids("test instruction")
    prepared = executor.prepare_release_inputs(release, input_ids, count=2)
    calls = []

    def forbid_vision(*args, **kwargs):
        raise AssertionError("The vision tower must never run during released inference")

    def inspect(_module, args, kwargs):
        assert kwargs.get("pixel_values") is None
        assert kwargs.get("input_ids") is None
        calls.append({"positions": kwargs["position_ids"].clone(),
                      "cache": kwargs["cache_position"].clone(),
                      "length": kwargs["inputs_embeds"].shape[1]})

    executor.model.model.visual.forward = forbid_vision
    hook = executor.model.register_forward_pre_hook(inspect, with_kwargs=True)
    try:
        outputs = executor.generate_tokens(release, input_ids, seeds=[991, 228], max_new_tokens=3)
    finally:
        hook.remove()
    assert len(outputs) == 2
    assert len(calls) >= 2  # A real cached next-token step, not only prefill.
    assert calls[0]["length"] == input_ids.shape[1]
    assert calls[1]["length"] == 1
    assert torch.equal(calls[0]["positions"][1:], prepared["latent_position_ids"])
    expected = (calls[1]["cache"].reshape(1, 1, -1)
                + prepared["latent_rope_deltas"].reshape(1, -1, 1)).expand(3, -1, -1)
    assert torch.equal(calls[1]["positions"][1:], expected)
    assert all(c.logprob_mean == pytest.approx(-math.log(64)) for c in outputs)
    assert executor.model.model.rope_deltas is None


def test_real_tiny_qwen_per_candidate_seed_is_batch_independent(tiny_executor):
    executor = tiny_executor
    ids = executor.prompt_token_ids("hello")
    release = np.zeros((25, 256))
    together = executor.generate_tokens(release, ids, seeds=[991, 228], max_new_tokens=3)
    alone = executor.generate_tokens(release, ids, seeds=[991], max_new_tokens=3)
    assert together[0].token_ids == alone[0].token_ids
    assert together[0].logprob_mean == alone[0].logprob_mean
    assert executor.instruction_embeddings("hello").shape == (5, 32)


def test_real_tiny_qwen_does_not_return_fake_coordinates(tiny_executor):
    result, candidates = tiny_executor.predict_grounding(
        np.zeros((25, 256)), prompt_text="Return JSON x and y", scoring_text="click save", seeds=[991])
    assert result.coordinate is None
    assert result.feedback == 2.0
    assert not candidates[0].valid
