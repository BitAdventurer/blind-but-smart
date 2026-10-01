"""CLI contract fixtures, never benchmark measurements or model inference."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from gui_joint_control.cli import main
from gui_joint_control.evaluation import assert_disjoint_manifests, file_hash, slot_identifier
from gui_joint_control.tms import TMSSchedule


@pytest.mark.parametrize('flag,value', [('--see','999'), ('--meth','CB'),
                                      ('--out','elsewhere'), ('--res','last.pt')])
def test_training_cli_rejects_abbreviated_binding_overrides(monkeypatch, capsys, flag, value):
    from gui_joint_control import cli
    dispatched = []
    monkeypatch.setattr(cli, 'fit', dispatched.append)
    argv = ['train','--replay','replay.npz','--updates','1','--family-id','f1',
            '--seed','100','--method','H','--output','registered-output']
    with pytest.raises(SystemExit) as error:
        cli.main(argv + [flag,value])
    assert error.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert not dispatched


@pytest.mark.parametrize('handler,argv,abbreviation', [
    ('prepare_dataset_command', ['prepare-dataset','--records','r','--splits','s','--task','G','--output','o'], '--out'),
    ('summarize_run_command', ['summarize-run','--run-directory','r','--output','o'], '--out'),
    ('smoke', ['smoke','--output','o'], '--out'),
    ('collect', ['collect-grounding','--manifest','m','--family-id','f','--output','o'], '--out'),
    ('collect', ['collect-action','--manifest','m','--family-id','f','--output','o'], '--out'),
    ('fit_projection', ['fit-projection','--records','r','--output','o'], '--out'),
    ('fit_adapter', ['fit-adapter','--model','m','--projection','p','--records','r','--output','o'], '--out'),
    ('build_tms', ['build-tms','--manifest','m','--development-transcript','t','--development-run','r',
                   '--development-manifest','d','--selected-h-checkpoint','h','--family-id','f',
                   '--public-hash-seed','s','--population','test','--task','G','--output','o'], '--out'),
    ('protocol_command', ['protocol','--registry','r','--output-root','o'], '--output-r'),
])
def test_other_cli_commands_reject_abbreviated_options(monkeypatch, capsys, handler, argv, abbreviation):
    from gui_joint_control import cli
    dispatched = []
    monkeypatch.setattr(cli, handler, dispatched.append)
    with pytest.raises(SystemExit) as error:
        cli.main(argv + [abbreviation,'elsewhere'])
    assert error.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert not dispatched


def test_split_guard_rejects_same_trajectory_different_slot(tmp_path):
    path = tmp_path/'dev.jsonl'
    path.write_text(json.dumps({'trajectory_id':'shared','slot':1})+'\n', encoding='utf-8')
    replay = SimpleNamespace(metadata={'slot_id':np.array([slot_identifier('shared',0)])})
    with pytest.raises(ValueError, match='share trajectories'):
        assert_disjoint_manifests(replay,path,task='G')


def tms_fixture(tmp_path):
    manifest = tmp_path/'population.jsonl'
    manifest.write_text(json.dumps({'trajectory_id':'target','slot':0})+'\n', encoding='utf-8')
    development = tmp_path/'development.jsonl'
    # Omitted eligible defaults to true for present rows.
    development.write_text(json.dumps({'trajectory_id':'dev','slot':0})+'\n', encoding='utf-8')
    checkpoint = tmp_path/'selected.pt'; checkpoint.write_bytes(b'synthetic checkpoint identity only')
    trace = tmp_path/'trace.jsonl'
    rows = [{'trajectory_id':'dev','slot':s,'eligible':s==0,'invoked':s==0,
             'correct':s==0,'candidate_count':2 if s==0 else 0,
             'executed_budgets':[2.0]*25 if s==0 else None} for s in range(56)]
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    provenance = {'kind':'new_recorded_controller_evaluation','split':'development','controller_method':'H',
                  'task':'G','family_id':'f1','manifest_sha256':file_hash(development),
                  'controller_checkpoint_sha256':file_hash(checkpoint),'transcript_sha256':file_hash(trace),
                  'selected_iteration':10000}
    selection = {'best':{'method':'H','iteration':10000,'checkpoint_sha256':file_hash(checkpoint)},
                 'binding':{'dev_manifest_sha256':file_hash(development),'task':'G','family_id':'f1'}}
    (tmp_path/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
    record = tmp_path/'run.json';record.write_text(json.dumps(provenance),encoding='utf-8')
    output = tmp_path/'schedule.json'
    argv = ['build-tms','--manifest',str(manifest),'--development-manifest',str(development),
            '--development-transcript',str(trace),'--development-run',str(record),
            '--selected-h-checkpoint',str(checkpoint),'--family-id','f1','--task','G',
            '--population','test','--public-hash-seed','a'*64,'--output',str(output)]
    return argv, record, output, trace


def test_tms_cli_binds_selected_h_trace_and_default_eligibility(tmp_path):
    argv, _, output, _ = tms_fixture(tmp_path)
    main(argv)
    schedule = TMSSchedule.from_json(output)
    budgets, counts = schedule.proposals([slot_identifier('target',0)])
    assert budgets.tolist()==[[2.]*25] and counts.tolist()==[2]
    with pytest.raises(FileExistsError):
        main(argv)


@pytest.mark.parametrize('field,value',[('controller_method','CB'),('split','test'),('family_id','f2'),
                                      ('controller_checkpoint_sha256','f'*64),('transcript_sha256','f'*64)])
def test_tms_cli_refuses_relabelled_or_changed_trace(tmp_path,field,value):
    argv, record, output, _ = tms_fixture(tmp_path)
    binding = json.loads(record.read_text(encoding='utf-8'));binding[field]=value
    record.write_text(json.dumps(binding),encoding='utf-8')
    with pytest.raises(ValueError,match='selected H'):
        main(argv)
    assert not output.exists()


def test_tms_cli_requires_complete_grid_even_when_hash_matches(tmp_path):
    argv, record, output, trace = tms_fixture(tmp_path)
    trace.write_text(trace.read_text(encoding='utf-8').splitlines()[0]+'\n',encoding='utf-8')
    binding=json.loads(record.read_text(encoding='utf-8'));binding['transcript_sha256']=file_hash(trace)
    record.write_text(json.dumps(binding),encoding='utf-8')
    with pytest.raises(ValueError,match='complete manifest'):
        main(argv)
    assert not output.exists()


def test_tms_cli_refuses_unselected_last_checkpoint(tmp_path):
    argv, _, output, _ = tms_fixture(tmp_path)
    path=tmp_path/'selection.json'
    state=json.loads(path.read_text(encoding='utf-8'))
    state['best']['checkpoint_sha256']='f'*64
    path.write_text(json.dumps(state),encoding='utf-8')
    with pytest.raises(ValueError,match='persisted development-selected'):
        main(argv)
    assert not output.exists()
