"""Runnable collection, offline fitting, checkpoints and software smoke tests."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import uuid
from dataclasses import asdict
import numpy as np


def load_config(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    from .controller import model_spec
    model_spec('H',config)
    return config


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def versions():
    result={'python':platform.python_version()}
    for name in ('torch','numpy','scipy','transformers'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def new_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def candidate_seed(seed, trajectory_id, slot, candidate_index):
    """Public decoding stream keyed independently of earlier candidate counts."""
    key=json.dumps(['decoder-v1',int(seed),str(trajectory_id),int(slot),int(candidate_index)],
        ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big') % (2**63-1)


def fit(args):
    from .replay import ReplayBuffer
    from .trainer import Trainer
    from .controller import describe_controllers
    config=load_config(args.config)
    buffer=ReplayBuffer.from_npz(args.replay)
    trainer=Trainer(config,args.method,buffer,seed=args.seed,device=args.device)
    if args.resume:trainer.load_checkpoint(args.resume)
    output=new_output(args.output)
    write_json(output/'config.json',config)
    write_json(output/'modules.json',describe_controllers(trainer.bundles))
    start=time.perf_counter()
    with (output/'training.jsonl').open('w',encoding='utf-8') as stream:
        for _ in range(args.updates):
            metrics=trainer.step(batch_size=args.batch_size)
            stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            stream.flush()
    trainer.save_checkpoint(output/'checkpoint.pt')
    report={'run_id':args.run_id or str(uuid.uuid4()),'kind':'new_offline_controller_fit','method':args.method,
        'updates_this_invocation':args.updates,'component_updates':trainer.component_updates,
        'replay':buffer.describe(),'runtime':versions(),'wall_seconds':time.perf_counter()-start,
        'config_sha256':hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),
        'checkpoint_selection':'last update; no held-out closed-loop development evaluation performed',
        'benchmark_accuracy_computed':False,'reproduces_historical_results':False}
    write_json(output/'run.json',report)
    print(json.dumps(report,indent=2))


def smoke(args):
    """Deliberately synthetic unit fixture, NEVER a paper result."""
    from .privacy import PrivacyLedger
    from .runtime import Slot,Prediction,behavior_allocator,run_trajectory,save_replay
    output=new_output(args.output)
    slots=[Slot('hit' if i%2==0 else 'miss',(.4,.4,.6,.6)) for i in range(8)]
    rng=np.random.default_rng(71)
    features=rng.normal(size=(8,25,256))
    ledger=PrivacyLedger([True]*8,probe_rng=np.random.default_rng(81),refinement_rng=np.random.default_rng(82))
    def executor(release,text,k):
        assert release.shape==(25,256) and 1<=k<=20
        return Prediction((.5,.5) if text=='hit' else (.1,.1),.2)
    records,transitions=run_trajectory(slots,lambda t:features[t],behavior_allocator(np.random.default_rng(91)),executor,ledger=ledger)
    assert len(records)==56
    save_replay(transitions,output/'software_fixture_replay.npz')
    config=load_config(args.config)
    config['reference_controller']['hidden_sizes']=[16,16]
    write_json(output/'software_fixture_config.json',config)
    nested=argparse.Namespace(config=output/'software_fixture_config.json',replay=output/'software_fixture_replay.npz',method='H',
        seed=123,device='cpu',resume=None,output=output/'fit',updates=args.updates,batch_size=8,run_id='software-smoke-'+str(uuid.uuid4()))
    fit(nested)
    report={'kind':'synthetic_software_test_only','published_experimental_evidence':False,
        'transcript_slots':len(records),'eligible_slots':8,'invocations':sum(r['invoked'] for r in records),
        'used_budget':ledger.used_budget,'cap':ledger.cap,'finite_training_steps':args.updates,
        'checkpoint_exists':(output/'fit/checkpoint.pt').is_file()}
    write_json(output/'SMOKE_ONLY.json',report)
    print(json.dumps(report,indent=2))


def manifest_rows(path):
    groups={}
    for line_number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        trajectory=str(row['trajectory_id']);slot=row['slot']
        if isinstance(slot,bool) or not isinstance(slot,int) or not 0<=slot<56:
            raise ValueError(f'Manifest line {line_number}: slot must be integer 0..55')
        if row.get('task','G')!='G':
            raise ValueError('This collection command supports Grounding; Action needs its separately pinned evaluator/schema')
        group=groups.setdefault(trajectory,{})
        if slot in group:raise ValueError('Duplicate trajectory/slot key')
        group[slot]=row
    if not groups:raise ValueError('Empty manifest')
    return groups


def fit_projection(args):
    from .fitting import initialize_projection,Stage1Trainer,AlignmentExample
    if Path(args.output).exists():raise FileExistsError(args.output)
    with np.load(args.records,allow_pickle=False) as archive:
        features=archive['features'];targets=archive['native_target_tokens']
    if features.ndim!=3 or features.shape[1:]!=(25,256) or targets.ndim!=3 or targets.shape[:2]!=features.shape[:2]:
        raise ValueError('Stage1 requires features[n,25,256] and native_target_tokens[n,25,hidden]')
    projection=initialize_projection(targets.shape[-1],seed=args.seed,device=args.device)
    trainer=Stage1Trainer(projection,seed=args.seed)
    records=[AlignmentExample(str(i),feature,target) for i,(feature,target) in enumerate(zip(features,targets))]
    history=trainer.fit(records)
    trainer.save(args.output,provenance={'records_sha256':hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),'kind':'new_alignment_fit'})
    write_json(Path(args.output)/'losses.json',history)
    print(json.dumps({'output':args.output,'screens':len(records),'epochs':2,'historical_result_reproduced':False}))


def fit_adapter(args):
    import torch
    from .executor import released_qwen_class,_immutable_model_ref
    from .fitting import initialize_projection,Stage2Trainer,TeacherForcedExample
    if Path(args.output).exists():raise FileExistsError(args.output)
    _immutable_model_ref(args.model,args.revision)
    native=released_qwen_class().from_pretrained(args.model,revision=args.revision,local_files_only=not args.allow_download,
        trust_remote_code=False,torch_dtype=getattr(torch,args.dtype)).to(args.device)
    matrix=np.load(args.projection,allow_pickle=False)
    projection=initialize_projection(matrix.shape[0],seed=args.seed,device=args.device)
    with torch.no_grad():projection.weight.copy_(torch.as_tensor(matrix,dtype=projection.weight.dtype,device=args.device))
    base=Path(args.records).resolve().parent
    records=[]
    for line in Path(args.records).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        records.append(TeacherForcedExample(str(row['record_id']),np.load(base/row['features_path'],allow_pickle=False),
            tuple(row['prompt_token_ids']),tuple(row['target_token_ids'])))
    trainer=Stage2Trainer(native,projection,seed=args.seed)
    history=trainer.fit(records)
    trainer.save(args.output,provenance={'records_sha256':hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
        'base_model':args.model,'revision':args.revision,'dtype':args.dtype,'kind':'new_task_adapter_fit'})
    write_json(Path(args.output)/'losses.json',history)
    if args.save_merged:
        model,_=trainer.merge_for_inference()
        model.save_pretrained(Path(args.output)/'merged_model',safe_serialization=True)
    print(json.dumps({'output':args.output,'screens':len(records),'epochs':3,'merged_model_saved':args.save_merged,'historical_result_reproduced':False}))


def collect(args):
    from .features import DinoRegionalEncoder
    from .executor import ReleasedQwenExecutor
    from .runtime import Slot,Prediction,behavior_allocator,inference_allocator,run_trajectory,save_replay
    from .privacy import PrivacyLedger
    from .prompt_policy import prepare_prompt
    groups=manifest_rows(args.manifest)
    trained_allocator=None
    if args.controller_checkpoint:
        if not args.training_replay or not args.controller_config:
            raise ValueError('Controller evaluation requires --training-replay and --controller-config for checkpoint binding')
        from .trainer import Trainer
        from .replay import ReplayBuffer
        trainer=Trainer(load_config(args.controller_config),args.controller_method,
            ReplayBuffer.from_npz(args.training_replay),device=args.device)
        trainer.load_checkpoint(args.controller_checkpoint)
        trained_allocator=inference_allocator(trainer)
    base=Path(args.manifest).resolve().parent
    projection=np.load(args.projection,allow_pickle=False)
    model=ReleasedQwenExecutor.from_pretrained(args.model,revision=args.revision,tokenizer_id=args.tokenizer or args.model,
        tokenizer_revision=args.tokenizer_revision or args.revision,projection=projection,device=args.device,
        dtype=args.dtype,local_files_only=not args.allow_download)
    # Cached feature files are permitted on the trusted client. No clean
    # feature/target object is passed into ReleasedQwenExecutor.
    encoder=None
    if any('image_path' in row and 'features_path' not in row for group in groups.values() for row in group.values()):
        if not args.public_projection or not args.dino_model:
            raise ValueError('Image manifests require --dino-model and --public-projection')
        encoder=DinoRegionalEncoder.from_pretrained(args.dino_model,args.dino_revision,args.public_projection,args.device)
    output=new_output(args.output)
    all_transitions=[];total_correct=0;eligible=0;invoked=0
    behavior_seed=np.random.SeedSequence(args.seed).spawn(1)[0]
    behavior_rng=np.random.default_rng(behavior_seed)
    (output/'releases').mkdir()
    format_wrapper='Return only a JSON object with normalized coordinates: {"x": number, "y": number}.\nInstruction: '
    with (output/'transcript.jsonl').open('w',encoding='utf-8') as stream:
        for trajectory,group in sorted(groups.items()):
            active_slot={}
            slots=[]
            for t in range(max(group)+1):
                row=group.get(t)
                slots.append(Slot('',None,False,False) if row is None else Slot(row.get('instruction',''),
                    tuple(row['target_box']) if row.get('target_box') is not None else None,
                    row.get('eligible',True),True))
            def load_features(t):
                active_slot['index']=t
                row=group[t]
                if 'features_path' in row:
                    return np.load(base/row['features_path'],allow_pickle=False)
                return encoder(base/row['image_path'])
            def execute(release,text,k):
                prepared=prepare_prompt('G',text,[],[],lambda payload:model.prompt_token_ids(format_wrapper+payload.current).reshape(-1).tolist())
                slot_index=active_slot['index']
                seeds=[candidate_seed(args.seed,trajectory,slot_index,i) for i in range(k)]
                result,candidates=model.predict_grounding(release,prompt_text=format_wrapper+prepared.payload.current,scoring_text=prepared.payload.scoring_text,seeds=seeds)
                release_name=hashlib.sha256(f'{trajectory}:{slot_index}'.encode()).hexdigest()+'.npy'
                np.save(output/'releases'/release_name,release,allow_pickle=False)
                with (output/'candidates.jsonl').open('a',encoding='utf-8') as candidate_stream:
                    candidate_stream.write(json.dumps({'trajectory_id':trajectory,'slot':slot_index,
                        'release_file':'releases/'+release_name,'public_decoder_seeds':seeds,
                        'input_ids':prepared.input_ids,'scoring_text':prepared.payload.scoring_text,
                        'scoring_token_ids':model.tokenizer.encode(prepared.payload.scoring_text,add_special_tokens=False),
                        'selected_index':result.selected_index,'candidates':[asdict(candidate) for candidate in candidates]},allow_nan=False)+'\n')
                return Prediction(result.coordinate,result.feedback)
            records,transitions=run_trajectory(slots,load_features,trained_allocator or behavior_allocator(behavior_rng),execute)
            for record in records:
                stream.write(json.dumps({'trajectory_id':trajectory,**record},allow_nan=False)+'\n')
            all_transitions.extend(transitions)
            eligible+=sum(r['eligible'] for r in records);invoked+=sum(r['invoked'] for r in records);total_correct+=sum(r['correct'] for r in records)
    if not trained_allocator:save_replay(all_transitions,output/'replay.npz')
    report={'kind':'new_grounding_controller_evaluation' if trained_allocator else 'new_grounding_behavior_collection','run_id':str(uuid.uuid4()),'eligible':eligible,'invoked':invoked,
        'correct':total_correct,'accuracy':total_correct/eligible if eligible else None,'runtime':versions(),
        'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'projection_sha256':hashlib.sha256(Path(args.projection).read_bytes()).hexdigest(),
        'executor':model.provenance,'public_seed':args.seed,'decoder_seed_scheme':'sha256-v1: seed/trajectory/slot/candidate',
        'model_dtype':args.dtype,'reproduces_historical_results':False}
    write_json(output/'run.json',report)
    print(json.dumps(report,indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    config_default=str(Path(__file__).resolve().parents[2]/'configs/naacl_reference.json')
    train=commands.add_parser('train',help='Fit real controller parameters from a validated immutable replay')
    train.add_argument('--config',default=config_default);train.add_argument('--replay',required=True)
    train.add_argument('--method',choices=['H','CB','Independent','Independent-1M'],default='H')
    train.add_argument('--updates',type=int,required=True);train.add_argument('--batch-size',type=int,default=256)
    train.add_argument('--device',default='cpu');train.add_argument('--seed',type=int,default=20260916)
    train.add_argument('--resume');train.add_argument('--run-id');train.add_argument('--output',required=True)
    train.set_defaults(func=fit)
    check=commands.add_parser('smoke',help='Run synthetic software checks, not a benchmark experiment')
    check.add_argument('--config',default=config_default);check.add_argument('--output',required=True)
    check.add_argument('--updates',type=int,default=3);check.set_defaults(func=smoke)
    collection=commands.add_parser('collect-grounding',help='Execute real released-latent VLM behavior collection')
    for field in ('manifest','model','projection','output'):collection.add_argument('--'+field,required=True)
    for field in ('revision','tokenizer','tokenizer-revision','dino-model','dino-revision','public-projection'):collection.add_argument('--'+field)
    collection.add_argument('--device',default='cpu');collection.add_argument('--seed',type=int,default=20260916)
    collection.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    collection.add_argument('--controller-checkpoint');collection.add_argument('--controller-config');collection.add_argument('--training-replay')
    collection.add_argument('--controller-method',choices=['H','CB','Independent','Independent-1M'],default='H')
    collection.add_argument('--allow-download',action='store_true');collection.set_defaults(func=collect)
    align=commands.add_parser('fit-projection',help='Train Stage1 against saved native visual tokens')
    align.add_argument('--records',required=True);align.add_argument('--output',required=True)
    align.add_argument('--device',default='cpu');align.add_argument('--seed',type=int,default=20260916);align.set_defaults(func=fit_projection)
    adapter=commands.add_parser('fit-adapter',help='Stage2 language q/v LoRA and projection teacher forcing')
    for field in ('model','projection','records','output'):adapter.add_argument('--'+field,required=True)
    adapter.add_argument('--revision');adapter.add_argument('--device',default='cpu');adapter.add_argument('--seed',type=int,default=20260916)
    adapter.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
    adapter.add_argument('--allow-download',action='store_true');adapter.add_argument('--save-merged',action='store_true');adapter.set_defaults(func=fit_adapter)
    args=parser.parse_args(argv)
    if hasattr(args,'updates') and args.updates<1:parser.error('--updates must be positive')
    args.func(args)


if __name__=='__main__':main()
