"""Cross-command software fixtures: no model downloads or benchmark execution."""
import json
import numpy as np
import pytest

from gui_joint_control.cli import main, versions
from gui_joint_control.evaluation import RecordedEvaluator, file_hash, validate_manifest_split
from gui_joint_control.runtime import TrajectoryExecutionError
from test_evaluation_driver import FakeModel, arguments, fixtures, write_manifest


def test_prepare_collect_summary_commands_share_artifact_bindings(tmp_path, monkeypatch):
    manifest, rows = fixtures(tmp_path)
    rows = [{**rows[0], 'trajectory_id': role} for role in ('train', 'dev', 'test')]
    write_manifest(manifest, rows)
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({role: [role] for role in ('train', 'dev', 'test')}), encoding='utf-8')
    prepared = tmp_path / 'prepared'
    main(['prepare-dataset', '--records', str(manifest), '--splits', str(splits),
          '--output', str(prepared), '--task', 'G'])
    from gui_joint_control import evaluation
    monkeypatch.setattr(evaluation, 'RecordedEvaluator', lambda args: RecordedEvaluator(args, model=FakeModel()))
    result = tmp_path / 'result'
    main(['collect-grounding', '--manifest', str(prepared / 'test.jsonl'), '--output', str(result),
          '--family-id', 'fixture-1', '--split', 'test', '--disable-retrieval'])
    target = tmp_path / 'summary.json'
    main(['summarize-run', '--run-directory', str(result), '--output', str(target)])
    report = json.loads(target.read_text(encoding='utf-8'))
    assert report['counts']['eligible'] == report['counts']['correct'] == 1
    assert report['metrics']['accuracy'] == 1
    assert report['provenance']['artifact_bindings']['manifest_sha256'] == file_hash(prepared / 'test.jsonl')
    assert report['provenance']['prompt_policy']['token_limit'] == 4096
    assert report['provenance']['manifest_split']['declared_roles'] == ['test']
    assert not any(item.startswith('transcript.') for item in report['missing_information'])
    assert 'Locate the requested target' not in target.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        main(['summarize-run', '--run-directory', str(result), '--output', str(target)])
    # A prepared train population cannot acquire a test label by changing argv.
    with pytest.raises(ValueError, match='manifest split'):
        main(['collect-grounding', '--manifest', str(prepared / 'train.jsonl'),
              '--output', str(tmp_path / 'wrong-role'), '--family-id', 'fixture-1',
              '--split', 'test', '--disable-retrieval'])
    assert not (tmp_path / 'wrong-role/releases').exists()


@pytest.mark.parametrize('declared,requested,valid', [
    ('train', 'fit-train', True), ('dev', 'development', True), ('test', 'evaluation', True),
    ('train', 'test', False), ('test', 'development', False)])
def test_manifest_role_aliases(declared, requested, valid):
    groups = {'episode': {0: {'split': declared}}}
    if valid:
        assert validate_manifest_split(groups, requested)['undeclared_records'] == 0
    else:
        with pytest.raises(ValueError, match='manifest split'):
            validate_manifest_split(groups, requested)


def test_legacy_manifest_split_omission_is_recorded_and_mixed_roles_rejected():
    assert validate_manifest_split({'episode': {0: {}}}, 'test')['undeclared_records'] == 1
    with pytest.raises(ValueError, match='mix'):
        validate_manifest_split({'episode': {0: {'split': 'train'}, 1: {'split': 'test'}}}, None)


def test_abort_preserves_completed_trajectory_failed_release_and_bound_context(tmp_path, monkeypatch):
    manifest, rows = fixtures(tmp_path)
    write_manifest(manifest, [{**rows[0], 'trajectory_id': name} for name in ('first', 'second')])

    class FailingModel(FakeModel):
        def predict_grounding(self, release, **kwargs):
            if self.calls:
                raise RuntimeError('synthetic inference failure')
            return super().predict_grounding(release, **kwargs)

    from gui_joint_control import evaluation
    monkeypatch.setattr(evaluation, 'RecordedEvaluator', lambda args: RecordedEvaluator(args, model=FailingModel()))
    output = tmp_path / 'aborted'
    with pytest.raises(TrajectoryExecutionError):
        main(['collect-grounding', '--manifest', str(manifest), '--output', str(output),
              '--family-id', 'fixture-1', '--split', 'test', '--disable-retrieval'])
    aborted = json.loads((output / 'aborted.json').read_text(encoding='utf-8'))
    transcript = [json.loads(line) for line in (output / 'transcript.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(transcript) == 57  # complete first trajectory, failed first slot of second
    assert transcript[0]['correct'] and transcript[-1]['status'] == 'EXECUTION_ERROR'
    assert transcript[-1]['trajectory_id'] == aborted['failed_trajectory_id'] == 'second'
    assert aborted['failed_trajectory_used_budget'] == transcript[-1]['used_budget'] > 0
    assert aborted['manifest_sha256'] == file_hash(manifest)
    assert aborted['transcript_sha256'] == file_hash(output / 'transcript.jsonl')
    assert aborted['prompt_policy']['token_limit'] == 4096
    assert aborted['runtime']['device'] is None  # fixture has no real device
    assert not aborted['benchmark_accuracy_computed'] and not (output / 'run.json').exists()
    assert len(list((output / 'releases').glob('*.npy'))) == 2
    assert (output / 'trusted-inputs.json').exists() and not (output / 'replay.npz').exists()
    from gui_joint_control.reporting import build_summary
    with pytest.raises(ValueError, match='aborted'):
        build_summary(output)


def test_environment_records_actual_cpu_precision_without_host_identity():
    result = versions(device='cpu', dtype='torch.float32')
    assert result['device'] == 'cpu' and result['dtype'] == 'float32'
    assert result['python'] and result['os'] and result['unicode']
    assert result['gpu_name'] is None
    assert not {'hostname', 'user', 'home', 'cwd'}.intersection(result)


@pytest.mark.parametrize('failure_slot', [0, 1])
@pytest.mark.parametrize('failure_kind,stage', [
    ('missing_input', 'feature_input'), ('invalid_features', 'release'),
    ('allocator', 'allocation'), ('invalid_proposal', 'release')])
def test_ledger_failure_preserves_completed_releases_without_inventing_invocation(
        tmp_path, monkeypatch, failure_slot, failure_kind, stage):
    from gui_joint_control import evaluation, runtime
    manifest, rows = fixtures(tmp_path)
    if failure_kind == 'missing_input':
        rows[failure_slot]['features_path'] = 'missing.npy'
    elif failure_kind == 'invalid_features':
        np.save(tmp_path / 'invalid.npy', np.full((25, 256), np.nan), allow_pickle=False)
        rows[failure_slot]['features_path'] = 'invalid.npy'
    write_manifest(manifest, rows)
    attempts = []
    def allocate(_observation):
        index = len(attempts)
        attempts.append(index)
        if index == failure_slot:
            if failure_kind == 'allocator':
                raise ValueError('private-probe-data must not enter persisted error text')
            if failure_kind == 'invalid_proposal':
                return np.full(25, np.nan), 1
        return np.full(25, 3.), 1
    monkeypatch.setattr(runtime, 'behavior_allocator', lambda rng: allocate)
    monkeypatch.setattr(evaluation, 'RecordedEvaluator', lambda args: RecordedEvaluator(args, model=FakeModel()))
    output = tmp_path / 'aborted'
    with pytest.raises(TrajectoryExecutionError) as failure:
        main(['collect-grounding', '--manifest', str(manifest), '--output', str(output),
              '--family-id', 'fixture-1', '--split', 'test', '--disable-retrieval'])
    aborted = json.loads((output / 'aborted.json').read_text(encoding='utf-8'))
    transcript = [json.loads(line) for line in (output / 'transcript.jsonl').read_text(encoding='utf-8').splitlines()]
    assert failure.value.__cause__ is not None
    assert aborted['failure_stage'] == stage and aborted['failed_slot'] == failure_slot
    assert not aborted['failed_slot_invoked']
    assert aborted['failed_trajectory_used_budget'] == 75. * failure_slot
    assert len(transcript) == failure_slot
    assert all(row['invoked'] and row['slot'] < failure_slot for row in transcript)
    assert len(list((output / 'releases').glob('*.npy'))) == failure_slot
    assert aborted['status'] == ('aborted_after_release' if failure_slot else 'aborted_before_release')
    assert aborted['manifest_sha256'] == file_hash(manifest)
    assert aborted['transcript_sha256'] == file_hash(output / 'transcript.jsonl')
    assert 'private-probe-data' not in (output / 'aborted.json').read_text(encoding='utf-8')
    assert (output / 'trusted-inputs.json').exists()
    assert not (output / 'run.json').exists() and not (output / 'replay.npz').exists()
    from gui_joint_control.reporting import build_summary
    with pytest.raises(ValueError, match='aborted'):
        build_summary(output)


def test_ledger_failure_on_later_trajectory_retains_prior_complete_trace(tmp_path, monkeypatch):
    from gui_joint_control import evaluation
    manifest, rows = fixtures(tmp_path)
    write_manifest(manifest, [{**rows[0], 'trajectory_id': 'first'},
                             {**rows[0], 'trajectory_id': 'second', 'features_path': 'missing.npy'}])
    monkeypatch.setattr(evaluation, 'RecordedEvaluator', lambda args: RecordedEvaluator(args, model=FakeModel()))
    output = tmp_path / 'aborted'
    with pytest.raises(TrajectoryExecutionError):
        main(['collect-grounding', '--manifest', str(manifest), '--output', str(output),
              '--family-id', 'fixture-1', '--split', 'test', '--disable-retrieval'])
    aborted = json.loads((output / 'aborted.json').read_text(encoding='utf-8'))
    assert aborted['status'] == 'aborted_after_release'
    assert aborted['failed_trajectory_id'] == 'second' and aborted['failed_slot'] == 0
    assert aborted['failed_trajectory_used_budget'] == 0 and not aborted['failed_slot_invoked']
    assert len(aborted['partial_records']) == 56
    assert {row['trajectory_id'] for row in aborted['partial_records']} == {'first'}
    assert sum(row['invoked'] for row in aborted['partial_records']) == 1
    assert aborted['partial_records'][-1]['used_budget'] > 0


def test_manifest_mutation_cannot_rebind_executed_population(tmp_path):
    manifest, rows = fixtures(tmp_path)

    class MutatingModel(FakeModel):
        def predict_grounding(self, release, **kwargs):
            result = super().predict_grounding(release, **kwargs)
            manifest.write_text(manifest.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            return result

    evaluator = RecordedEvaluator(arguments(), model=MutatingModel())
    output = tmp_path / 'changed'
    with pytest.raises(ValueError, match='manifest changed'):
        evaluator.run(manifest, output=output, split='test')
    assert len((output / 'transcript.jsonl').read_text(encoding='utf-8').splitlines()) == 56
    assert not (output / 'replay.npz').exists() and not (output / 'run.json').exists()
