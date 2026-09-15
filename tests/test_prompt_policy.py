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
    assert result.removed_history==2 and result.removed_retrieval==1
    assert result.payload.retrieval==(Retrieval('HIGH',.9),)
    assert result.payload.scoring_text==current
    assert len(result.input_ids)==14


def test_tied_retrieval_drops_last_and_grounding_excludes_history():
    result=prepare_prompt('G','x',['ignored'],[Retrieval('a',.2),Retrieval('b',.2)],renderer,token_limit=7)
    assert result.payload.history==()
    assert result.payload.retrieval==(Retrieval('a',.2),)
    assert result.payload.scoring_text=='x'


def test_fixed_prompt_overflow_aborts_instead_of_silently_excluding_example():
    with pytest.raises(PromptTooLong):
        prepare_prompt('G','instruction',[],[],renderer,token_limit=3)
