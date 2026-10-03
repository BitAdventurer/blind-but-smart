"""Preparation-to-fit CLI wiring with tiny local arrays and no model downloads."""
import json
import numpy as np
import pytest

from gui_joint_control.cli import main
from test_training_data import Tokenizer


def fixture(tmp_path):
    np.save(tmp_path/'features.npy',np.eye(25,256)*.5)
    np.save(tmp_path/'targets.npy',np.eye(25,32,dtype=np.float32))
    row={'task':'G','split':'train','trajectory_id':'t1','slot':0,'eligible':True,
         'instruction':'Click Save','features_path':'features.npy',
         'native_target_tokens_path':'targets.npy','target_text':'{"x":0.2,"y":0.4}'}
    source=tmp_path/'train.jsonl';source.write_text(json.dumps(row)+'\n',encoding='utf-8')
    return source


def test_cli_alignment_preparation_then_actual_tiny_fitting(tmp_path):
    source=fixture(tmp_path)
    main(['prepare-alignment','--manifest',str(source),'--output',str(tmp_path/'prepared')])
    main(['fit-projection','--records',str(tmp_path/'prepared/alignment.npz'),'--output',str(tmp_path/'fit')])
    report=json.loads((tmp_path/'fit/fit_metadata.json').read_text())
    assert report['provenance']['prepared_input_binding']['stage']==1
    assert np.load(tmp_path/'fit/projection.npy').shape==(32,256)


def test_cli_supervised_preparation_and_tamper_rejected_before_model_load(tmp_path,monkeypatch):
    from gui_joint_control import training_data, executor
    source=fixture(tmp_path)
    monkeypatch.setattr(training_data,'load_training_tokenizer',lambda *a,**k:Tokenizer())
    main(['prepare-supervised','--manifest',str(source),'--task','G','--tokenizer','fixture-tokenizer',
          '--output',str(tmp_path/'prepared')])
    records=tmp_path/'prepared/records.jsonl'
    row=json.loads(records.read_text())
    assert row['target_token_ids'][-1]==63
    np.save(tmp_path/'prepared'/row['features_path'],np.eye(25,256)*.25)
    calls=[]
    monkeypatch.setattr(executor,'released_qwen_class',lambda:calls.append('model-loaded'))
    with pytest.raises(ValueError):
        main(['fit-adapter','--records',str(records),'--model','unused','--projection','unused.npy',
              '--output',str(tmp_path/'fit')])
    assert not calls and not (tmp_path/'fit').exists()


def test_cli_accepts_actor_tms_method_without_running_training(monkeypatch):
    from gui_joint_control import cli
    calls=[]
    monkeypatch.setattr(cli,'fit',calls.append)
    main(['train','--method','H-ActorTMS','--replay','r','--updates','1','--family-id','f1',
          '--tms-schedule','s','--output','unused'])
    assert calls[0].method=='H-ActorTMS' and calls[0].tms_schedule=='s'


def test_dataset_prepare_rebases_optional_native_targets(tmp_path):
    source=fixture(tmp_path)
    row=json.loads(source.read_text());row['target_box']=[.1,.1,.5,.5]
    source.write_text(json.dumps(row)+'\n')
    splits=tmp_path/'splits.json';splits.write_text(json.dumps({'train':['t1'],'dev':[],'test':[]}))
    output=tmp_path/'prepared'
    main(['prepare-dataset','--records',str(source),'--splits',str(splits),'--task','G','--output',str(output)])
    main(['prepare-alignment','--manifest',str(output/'train.jsonl'),'--output',str(tmp_path/'aligned')])
    assert np.load(tmp_path/'aligned/alignment.npz')['native_target_tokens'].shape==(1,25,32)
