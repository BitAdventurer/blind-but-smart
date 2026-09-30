"""Full-input cap and exact scoring-span behavior using a test renderer."""
import pytest
from gui_joint_control.prompt_policy import Retrieval,prepare_prompt,PromptTooLong


def renderer(payload):
    # Token IDs for a software fixture, not the production tokenizer.
    return [42]* (5 + len(payload.current) + sum(map(len,payload.history)) + sum(len(x.text) for x in payload.retrieval))


def test_old_history_removed_before_low_score_retrieval_and_span_is_exact():
    current='  요청\n'
    result=prepare_prompt('A',current,['older','recent'],[Retrieval('HIGH',.9),Retrieval('low',.1)],renderer,token_limit=14)
    assert result.payload.current==current
    assert result.payload.history==()
    assert result.retained_history_positions==()
    assert result.removed_history==2 and result.removed_retrieval==1
    assert result.payload.retrieval==(Retrieval('HIGH',.9),)
    assert result.payload.scoring_text==current
    assert len(result.input_ids)==14


def test_tied_retrieval_drops_last_and_grounding_excludes_history():
    result=prepare_prompt('G','x',['ignored'],[Retrieval('a',.2),Retrieval('b',.2)],renderer,token_limit=7)
    assert result.payload.history==()
    assert result.payload.retrieval==(Retrieval('a',.2),)
    assert result.payload.scoring_text=='x'
    assert result.retained_history_positions==()


def test_fixed_prompt_overflow_aborts_instead_of_silently_excluding_example():
    with pytest.raises(PromptTooLong):
        prepare_prompt('G','instruction',[],[],renderer,token_limit=3)


@pytest.mark.parametrize("count, max_steps, cap, expected", [
    (12, 10, 9, (9, 10, 11)),
    (4, 2, 100, (2, 3)),
    (3, 0, 100, ()),
])
def test_history_positions_preserve_original_indices_after_tail_and_overflow(count, max_steps, cap, expected):
    history = ["x"] * count
    prepared = prepare_prompt("A", "y", history, [], renderer,
                              token_limit=cap, max_previous_steps=max_steps)
    assert prepared.retained_history_positions == expected
    assert prepared.payload.history == tuple(history[i] for i in expected)
    assert prepared.removed_history == count - len(expected)


def test_duplicate_retrieval_text_keeps_distinct_ids_without_changing_model_input():
    from gui_joint_control.evaluation import render_prompt
    entries = [Retrieval("same", score, (f"train-{i}", i, "A"))
               for i, score in enumerate((.9, .1, .9))]
    traced = prepare_prompt("A", "x", [], entries, renderer, token_limit=14)
    legacy = prepare_prompt("A", "x", [], [Retrieval(x.text, x.score) for x in entries],
                            renderer, token_limit=14)
    assert [x.source_id for x in traced.payload.retrieval] == [("train-0", 0, "A"), ("train-2", 2, "A")]
    assert [(x.text, x.score) for x in traced.payload.retrieval] == [(x.text, x.score) for x in legacy.payload.retrieval]
    assert traced.input_ids == legacy.input_ids
    assert traced.payload.scoring_text == legacy.payload.scoring_text
    assert render_prompt(traced.payload) == render_prompt(legacy.payload)
    assert traced.removed_retrieval == legacy.removed_retrieval == 1


@pytest.mark.parametrize("source_id", ["train", ("", 0, "G"), ("train", True, "G"),
                                       ("train", -1, "G"), ("train", 0, "X")])
def test_retrieval_source_identity_rejects_invalid_metadata(source_id):
    with pytest.raises(ValueError, match="source_id"):
        Retrieval("example", .9, source_id)


def test_retrieval_source_task_must_match_prompt():
    with pytest.raises(ValueError, match="source task"):
        prepare_prompt("G", "x", [], [Retrieval("example", .9, ("train", 0, "A"))], renderer)
