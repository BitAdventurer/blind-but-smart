"""Synthetic Action protocol tests, without model weights or benchmark data."""
import json
import numpy as np
import pytest

from gui_joint_control.action_evaluation import (ActionSlot, ActionPrediction,
                                               ActionScores, load_action_schemas)
from gui_joint_control.executor import ReleasedQwenExecutor, GeneratedCandidate
from gui_joint_control.runtime import run_action_trajectory, TrajectoryExecutionError
from gui_joint_control.scoring import ActionFunctionSchema, parse_action


def schema():
    return {"click": ActionFunctionSchema(frozenset({"point"}), frozenset({"label"}),
                                          frozenset({"point"}), frozenset({"ok"}),
                                          argument_types={"label": "string"})}


def test_explicit_train_alias_and_type_contract(tmp_path):
    payload = {"schema_id": "fixture-v1", "source_split": "fit-train", "source_manifest_sha256": "a" * 64,
               "functions": {"click": {"required": ["point"], "optional": ["label"], "spatial": ["point"],
                                        "statuses": ["ok"], "status_aliases": {"done": "ok"},
                                        "argument_types": {"label": "string"}}}, "function_aliases": {"tap": "click"}}
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    schemas = load_action_schemas(path)
    parsed = parse_action('{"function":"tap","arguments":{"point":[-0.0000001,1.0000001]},"status":"done"}', schemas)
    assert parsed == {"function": "click", "arguments": {"point": [0., 1.]}, "status": "ok"}
    assert parse_action('{"function":"click","arguments":{"point":[true,0]},"status":"ok"}', schemas) is None
    assert parse_action('{"function":"click","arguments":{"point":[0,0],"label":42},"status":"ok"}', schemas) is None
    assert parse_action('{"function":"unknown","arguments":{"point":[0,0]},"status":"ok"}', schemas) is None
    payload["source_split"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fit-train"):
        load_action_schemas(path)


def test_prediction_reuses_one_release_and_all_spatial_keys():
    # Test the complete library generation -> parse -> relevance -> aggregation
    # boundary using a fixed decoder. No tiny/random model is a paper result.
    executor = object.__new__(ReleasedQwenExecutor)
    executor.projection = np.eye(2, 256)
    executor.prompt_token_ids = lambda text: (1, 2)
    executor.instruction_embeddings = lambda text: np.array([[1., 0.]])
    release = np.zeros((25, 256)); release[:, 0] = 1
    calls = []
    def generate(received, ids, *, seeds, max_new_tokens):
        assert received is release
        calls.append((tuple(seeds), max_new_tokens))
        return (GeneratedCandidate(0, '{"function":"click","arguments":{"point":[0.1,0.2]},"status":"ok"}', (11,), -.2, True),
                GeneratedCandidate(1, "not json", (12,), -.5, True))
    executor.generate_tokens = generate
    action, feedback, candidates = executor.predict_action(release, prompt_text="public request",
                                                           scoring_text="public request", seeds=[7, 8], schemas=schema())
    assert action["function"] == "click" and feedback == 1.0
    assert len(calls) == 1 and calls[0] == ((7, 8), 128)
    assert candidates[0].relevance == pytest.approx(1.)
    assert not candidates[1].valid and candidates[1].text == "not json"


def test_action_runtime_reference_never_crosses_executor_and_feedback_is_prior():
    reference = {"function": "click", "arguments": {"point": [.5, .5]}, "status": "ok"}
    slots = [ActionSlot("current request", reference, ("older thought", "later thought")) for _ in range(2)]
    seen = []
    def execute(release, request, history, k, prior):
        assert request == "current request" and history == ("older thought", "later thought")
        assert release.shape == (25, 256) and k == 2
        seen.append(prior)
        # Independent object, not the offline reference object.
        return ActionPrediction({"function": "click", "arguments": {"point": [.5, .5]}, "status": "ok"}, .75)
    score_calls = []
    def evaluate(action, slot):
        score_calls.append(slot.reference_action)
        assert action is not slot.reference_action
        return ActionScores(True, True, len(score_calls) == 1)
    records, transitions = run_action_trajectory(slots, lambda t: np.zeros((25, 256)),
        lambda state: (np.full(25, 1.5), 2), execute, evaluator=evaluate)
    assert seen == [0., .75]
    assert sum(row["correct"] for row in records) == 1
    assert sum(row["function_correct"] for row in records) == 2
    assert transitions[0]["success"] is True and transitions[1]["success"] is False
    assert len(records) == 56 and len(transitions) == 2


def test_invalid_action_and_exhaustion_remain_component_misses():
    slots = [ActionSlot("request", {"function": "x"}) for _ in range(8)]
    reads = []
    def load(t):
        reads.append(t)
        return np.zeros((25, 256))
    records, transitions = run_action_trajectory(slots, load, lambda state: (np.full(25, 5.), 1),
        lambda *args: ActionPrediction({"function": "INVALID", "arguments": {}, "status": "INVALID"}, 2.),
        evaluator=lambda *args: pytest.fail("Invalid predictions never need the official scorer"))
    assert reads == list(range(6)) and sum(row["eligible"] for row in records) == 8
    assert not any(row["correct"] or row["function_correct"] for row in records)
    assert transitions[-1]["reward"] == pytest.approx(.5 - .99 - .99 ** 2)


def test_failure_reports_already_spent_release_and_aborts_run():
    def fail(*args):
        raise ValueError("mandatory prompt overflow")
    with pytest.raises(TrajectoryExecutionError) as failure:
        run_action_trajectory([ActionSlot("request", {"function": "x"})], lambda t: np.zeros((25, 256)),
            lambda state: (np.full(25, 1.5), 1), fail, evaluator=lambda *a: ActionScores(True, True, True))
    assert failure.value.used_budget == 37.5
    assert failure.value.records[-1]["status"] == "EXECUTION_ERROR"
    assert failure.value.records[-1]["invoked"]
