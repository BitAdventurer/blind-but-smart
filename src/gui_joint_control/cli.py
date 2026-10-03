"""Runnable collection, offline fitting, checkpoints and software smoke tests."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
from importlib.resources import files
import json
import platform
import time
import uuid
import unicodedata
from dataclasses import asdict
import numpy as np


def load_config(path=None):
    source = files('gui_joint_control').joinpath('configs/naacl_reference.json') if path is None else Path(path)
    config=json.loads(source.read_text(encoding='utf-8'))
    from .controller import model_spec
    model_spec('H',config)
    return config


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def versions(*, device=None, dtype=None):
    result={'python':platform.python_version(), 'os':platform.system(),
            'os_release':platform.release(), 'machine':platform.machine(),
            'unicode':unicodedata.unidata_version,
            'device':str(device) if device is not None else None,
            'dtype':str(dtype).removeprefix('torch.') if dtype is not None else None}
    for name in ('torch','numpy','scipy','transformers','peft','Pillow','accelerate','safetensors'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    import torch
    result.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu_name=None)
    if device is not None and torch.device(device).type == 'cuda':
        result['gpu_name'] = torch.cuda.get_device_name(torch.device(device))
    return result


def new_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def candidate_seed(seed, trajectory_id, slot, candidate_index):
    from .evaluation import decoder_seed
    return decoder_seed(seed, trajectory_id, slot, candidate_index)


def fit(args):
    from .replay import ReplayBuffer
    from .trainer import Trainer
    from .controller import describe_controllers
    from .selection import CheckpointManager
    from .evaluation import RecordedEvaluator, file_hash, assert_disjoint_manifests, validate_tms_population
    from .tms import TMSSchedule
    config = load_config(args.config)
    buffer = ReplayBuffer.from_npz(args.replay)
    software_only = getattr(args, 'software_only', False)
    if not software_only and args.batch_size != 256:
        raise ValueError('Manuscript fitting uses batch size 256; smaller batches are software fixtures only')
    task, family = getattr(args, 'task', 'G'), getattr(args, 'family_id', 'software-fixture')
    schedule = TMSSchedule.from_json(args.tms_schedule) if getattr(args, 'tms_schedule', None) else None
    dev_schedule = TMSSchedule.from_json(args.dev_tms_schedule) if getattr(args, 'dev_tms_schedule', None) else None
    if not software_only:
        for field, expected in (('task', task), ('family_id', family)):
            values = buffer.metadata.get(field)
            if values is None or values.shape != (len(buffer),) or not np.all(values == expected):
                raise ValueError(f'Replay {field} must bind the explicit fitting task/family')
        source = buffer.metadata.get('source_manifest_sha256')
        if source is None or source.shape != (len(buffer),) or len(set(source.tolist())) != 1:
            raise ValueError('Replay must bind one immutable source manifest SHA256')
        if schedule and schedule.population_manifest_sha256 != source[0]:
            raise ValueError('Training TMS population and replay source manifest differ')
    trainer = Trainer(config, args.method, buffer, seed=args.seed, device=args.device,
                      tms_schedule=schedule, task=task, family=family)
    manager = evaluator = None
    if not software_only:
        if not args.dev_manifest:
            raise ValueError('Manuscript fitting requires --dev-manifest; --software-only is only for software fixtures')
        assert_disjoint_manifests(buffer, args.dev_manifest, task=task)
        if args.method in ('Disclosure-only', 'Count-only') and dev_schedule is None:
            raise ValueError('Single-head fitting requires --dev-tms-schedule before optimization')
        if dev_schedule is not None:
            validate_tms_population(dev_schedule, args.dev_manifest, task=task, family=family)
        evaluator = RecordedEvaluator(args)
    if args.resume:
        if software_only:
            raise ValueError('Use a new output for software fixtures')
        if Path(args.resume).resolve() != (Path(args.output)/'last.pt').resolve():
            raise ValueError('Resume requires the same output directory and its last.pt selection state')
        output = Path(args.output)
        trainer.load_checkpoint(args.resume)
    else:
        output = new_output(args.output)
    write_json(output/'config.json', config)
    write_json(output/'modules.json', describe_controllers(trainer.bundles))
    if evaluator:
        binding = {'dev_manifest_sha256': file_hash(args.dev_manifest), 'task': task, 'family_id': family,
                   'method': args.method, 'replay_content_sha256': buffer.content_sha256,
                   'config_sha256': trainer.config_sha256, 'seed': args.seed,
                   'dev_replicates': args.dev_replicates, 'executor': evaluator.provenance,
                   'dev_tms_schedule_sha256': dev_schedule.content_sha256 if dev_schedule else None}
        manager = CheckpointManager(output, binding=binding, interval=10000, resume=bool(args.resume))
        if args.resume:
            manager.validate_resume(trainer)
    last_development = {}
    def evaluate(current):
        episodes = []
        for replicate in range(1, args.dev_replicates + 1):
            _, traces = evaluator.run(args.dev_manifest, trainer=current,
                replicate='development-' + str(replicate), tms_schedule=dev_schedule, split='development')
            episodes.extend(traces)
        last_development['episodes'] = episodes
        return {'split':'development', 'manifest_sha256':file_hash(args.dev_manifest), 'episodes':episodes}
    start = time.perf_counter()
    with (output/'training.jsonl').open('a' if args.resume else 'w', encoding='utf-8') as stream:
        for _ in range(args.updates):
            metrics = trainer.step(batch_size=args.batch_size)
            stream.write(json.dumps(metrics, allow_nan=False)+'\n')
            stream.flush()
            if manager:
                selection = manager.consider(trainer, evaluate)
                if selection and selection['selected']:
                    trace = output/'selected-development.jsonl'
                    with trace.open('w', encoding='utf-8') as records:
                        for episode in last_development['episodes']:
                            for row in episode['records']:
                                records.write(json.dumps({'trajectory_id':episode['trajectory_id'],
                                    'replicate_id':episode['replicate'], **row}, allow_nan=False)+'\n')
                    write_json(output/'selected-development-run.json', {
                        'kind':'new_recorded_controller_evaluation', 'split':'development',
                        'controller_method':args.method, 'task':task, 'family_id':family,
                        'controller_checkpoint_sha256':file_hash(manager.selected_path),
                        'manifest_sha256':file_hash(args.dev_manifest), 'transcript_sha256':file_hash(trace),
                        'selected_iteration':trainer.iteration, 'replicates':args.dev_replicates,
                        'score':selection['score'], **evaluator.provenance,
                        'runtime':versions(device=getattr(evaluator.model, 'device', None),
                                           dtype=getattr(evaluator.model, 'dtype', None)),
                        'selection_binding':binding, 'reproduces_historical_results':False})
    if manager:
        manager.save_last(trainer)
    else:
        trainer.save_checkpoint(output/'checkpoint.pt')
    report = {'run_id':args.run_id or str(uuid.uuid4()), 'kind':'software_fixture_fit' if software_only else 'new_offline_controller_fit',
        'method':args.method, 'family_id':family, 'task':task,
        'updates_this_invocation':args.updates, 'component_updates':trainer.component_updates,
        'replay':buffer.describe(), 'runtime':versions(device=trainer.device, dtype='float32'), 'wall_seconds':time.perf_counter()-start,
        'config_sha256':file_hash(output/'config.json'),
        'checkpoint_selection':manager.state['criterion'] if manager else 'software fixture last update only',
        'selected_checkpoint':str(manager.selected_path) if manager and manager.best else None,
        'selected_iteration':manager.best['iteration'] if manager and manager.best else None,
        'development_evaluations':len(manager.state['evaluations']) if manager else 0,
        'benchmark_accuracy_computed':False, 'reproduces_historical_results':False}
    write_json(output/'run.json', report)
    print(json.dumps(report, indent=2))


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
    write_json(output/'software_fixture_config.json',config)
    nested=argparse.Namespace(config=output/'software_fixture_config.json',replay=output/'software_fixture_replay.npz',method='H',
        seed=123,device='cpu',resume=None,output=output/'fit',updates=args.updates,batch_size=8,software_only=True,run_id='software-smoke-'+str(uuid.uuid4()))
    fit(nested)
    report={'kind':'synthetic_software_test_only','published_experimental_evidence':False,
        'transcript_slots':len(records),'eligible_slots':8,'invocations':sum(r['invoked'] for r in records),
        'used_budget':ledger.used_budget,'cap':ledger.cap,'finite_training_steps':args.updates,
        'checkpoint_exists':(output/'fit/checkpoint.pt').is_file()}
    write_json(output/'SMOKE_ONLY.json',report)
    print(json.dumps(report,indent=2))


def manifest_rows(path, task='G'):
    from .evaluation import manifest_rows as read_manifest
    return read_manifest(path, task)


def fit_projection(args):
    from .fitting import initialize_projection,Stage1Trainer,AlignmentExample
    from .training_data import verify_prepared_inputs
    if Path(args.output).exists():raise FileExistsError(args.output)
    prepared_binding = verify_prepared_inputs(args.records, stage=1)
    with np.load(args.records,allow_pickle=False) as archive:
        features=archive['features'];targets=archive['native_target_tokens']
    if features.ndim!=3 or features.shape[1:]!=(25,256) or targets.ndim!=3 or targets.shape[:2]!=features.shape[:2]:
        raise ValueError('Stage1 requires features[n,25,256] and native_target_tokens[n,25,hidden]')
    projection=initialize_projection(targets.shape[-1],seed=args.seed,device=args.device)
    trainer=Stage1Trainer(projection,seed=args.seed)
    records=[AlignmentExample(str(i),feature,target) for i,(feature,target) in enumerate(zip(features,targets))]
    history=trainer.fit(records)
    trainer.save(args.output,provenance={'records_sha256':hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
                                       'prepared_input_binding':prepared_binding,'kind':'new_alignment_fit'})
    write_json(Path(args.output)/'losses.json',history)
    print(json.dumps({'output':args.output,'screens':len(records),'epochs':2,'historical_result_reproduced':False}))


def fit_adapter(args):
    import torch
    from .executor import released_qwen_class,_immutable_model_ref
    from .fitting import initialize_projection,Stage2Trainer,TeacherForcedExample
    from .training_data import verify_prepared_inputs
    if Path(args.output).exists():raise FileExistsError(args.output)
    prepared_binding = verify_prepared_inputs(args.records, stage=2)
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
        'prepared_input_binding':prepared_binding,
        'base_model':args.model,'revision':args.revision,'dtype':args.dtype,'kind':'new_task_adapter_fit'})
    write_json(Path(args.output)/'losses.json',history)
    if args.save_merged:
        model,_=trainer.merge_for_inference()
        model.save_pretrained(Path(args.output)/'merged_model',safe_serialization=True)
    print(json.dumps({'output':args.output,'screens':len(records),'epochs':3,'merged_model_saved':args.save_merged,'historical_result_reproduced':False}))


def collect(args):
    from .trainer import Trainer
    from .replay import ReplayBuffer
    from .tms import TMSSchedule
    from .evaluation import RecordedEvaluator, file_hash
    from .runtime import TrajectoryExecutionError
    trainer = None
    training_schedule = TMSSchedule.from_json(args.tms_schedule) if args.tms_schedule else None
    evaluation_schedule = TMSSchedule.from_json(args.evaluation_tms_schedule) if args.evaluation_tms_schedule else None
    if args.controller_checkpoint and args.controller_method in ('Disclosure-only','Count-only') and evaluation_schedule is None:
        raise ValueError('Single-head evaluation requires --evaluation-tms-schedule before any release')
    if args.controller_checkpoint:
        if not args.training_replay or not args.controller_config:
            raise ValueError('Controller evaluation requires --training-replay and --controller-config')
        trainer = Trainer(load_config(args.controller_config), args.controller_method,
            ReplayBuffer.from_npz(args.training_replay), device=args.device, tms_schedule=training_schedule,
            task=args.task, family=args.family_id)
        trainer.load_checkpoint(args.controller_checkpoint)
    evaluator = RecordedEvaluator(args)
    output = new_output(args.output)
    binding = dict(kind='new_recorded_controller_evaluation' if trainer or evaluation_schedule else 'new_behavior_collection',
                  run_id=str(uuid.uuid4()),
                  runtime=versions(device=getattr(evaluator.model, 'device', None), dtype=getattr(evaluator.model, 'dtype', None)),
                  split=args.split or ('evaluation' if trainer or evaluation_schedule else 'fit-train'),
                  controller_method=args.controller_method if trainer else ('TMS' if evaluation_schedule else 'behavior'),
                  controller_checkpoint_sha256=file_hash(args.controller_checkpoint) if trainer else None,
                  controller_config_sha256=file_hash(args.controller_config) if trainer else None,
                  training_replay_sha256=file_hash(args.training_replay) if trainer else None,
                  tms_schedule_sha256=training_schedule.content_sha256 if training_schedule else None,
                  evaluation_tms_schedule_sha256=evaluation_schedule.content_sha256 if evaluation_schedule else None)
    try:
        report, _ = evaluator.run(args.manifest, trainer=trainer, replicate=args.replicate_id,
                                 output=output, tms_schedule=evaluation_schedule, split=binding['split'])
    except TrajectoryExecutionError as error:
        write_json(output/'aborted.json', {**binding, **error.evaluation_context,
            'status':'aborted_after_release' if any(row['invoked'] for row in error.records) else 'aborted_before_release',
            'scope':'trusted local only; not protected public mechanism output',
            'error':f'{error.failure_stage} failed ({error.error_type})', 'partial_records':error.records,
            'transcript_sha256':file_hash(output/'transcript.jsonl'),
            'benchmark_accuracy_computed':False, 'reproduces_historical_results':False})
        raise
    report.update(**binding, transcript_sha256=file_hash(output/'transcript.jsonl'))
    write_json(output/'run.json', report)
    print(json.dumps(report, indent=2))


METHODS = ['H','H-ActorTMS','CB','Independent','Independent-1M','Disclosure-only','Count-only']


def add_executor_arguments(parser):
    for name in ('model','projection','revision','tokenizer','tokenizer-revision','dino-model','dino-revision',
                 'public-projection','action-schema','action-evaluator','retrieval-bank','retrieval-keys','retrieval-exclusion-manifest'):
        parser.add_argument('--'+name)
    parser.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    parser.add_argument('--retrieval-threshold', type=float)
    parser.add_argument('--retrieval-view', choices=['primary','strict'], default='primary')
    parser.add_argument('--disable-retrieval', action='store_true', help='Explicit retrieval-disabled ablation')
    parser.add_argument('--allow-download', action='store_true')


def protocol_command(args):
    from .protocol import build_plan, execute_plan
    registry = json.loads(Path(args.registry).read_text(encoding='utf-8'))
    plan = build_plan(registry, args.output_root)
    if args.plan_output:
        write_json(args.plan_output, plan)
    if args.execute:
        execute_plan(plan)
    else:
        print(json.dumps(plan, indent=2))


def prepare_dataset_command(args):
    from .dataset_preparation import prepare_dataset
    report = prepare_dataset(args.records, args.splits, args.output, task=args.task, action_schema=args.action_schema)
    print(json.dumps(report, indent=2))


def summarize_run_command(args):
    from .reporting import summarize_run
    print(json.dumps(summarize_run(args.run_directory, args.output), indent=2))


def build_tms(args):
    from .tms import build_tms_schedule
    from .evaluation import file_hash, slot_identifier
    groups = manifest_rows(args.manifest, args.task)
    provenance = json.loads(Path(args.development_run).read_text(encoding='utf-8'))
    expected = {'kind':'new_recorded_controller_evaluation', 'split':'development', 'controller_method':'H',
                'task':args.task, 'family_id':args.family_id,
                'manifest_sha256':file_hash(args.development_manifest),
                'controller_checkpoint_sha256':file_hash(args.selected_h_checkpoint),
                'transcript_sha256':file_hash(args.development_transcript)}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('TMS requires the selected H development transcript with matching task/family/artifact provenance')
    selection_path = Path(args.selection_state) if args.selection_state else Path(args.selected_h_checkpoint).parent/'selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    best, binding = selection.get('best') or {}, selection.get('binding') or {}
    if (best.get('method') != 'H' or best.get('checkpoint_sha256') != expected['controller_checkpoint_sha256']
            or type(best.get('iteration')) is not int or best['iteration'] <= 0
            or provenance.get('selected_iteration') != best['iteration']
            or binding.get('dev_manifest_sha256') != expected['manifest_sha256']
            or binding.get('task') != args.task or binding.get('family_id') != args.family_id):
        raise ValueError('TMS checkpoint must match the persisted development-selected H, not an unselected last state')
    rows = [json.loads(line) for line in Path(args.development_transcript).read_text(encoding='utf-8').splitlines() if line.strip()]
    development_groups = manifest_rows(args.development_manifest, args.task)
    by_replicate = {}
    for row in rows:
        replicate = str(row.get('replicate_id', provenance.get('replicate_id', '1')))
        population = by_replicate.setdefault(replicate, {})
        key = (row['trajectory_id'], row['slot'])
        if key in population:
            raise ValueError('Duplicate development transcript row')
        population[key] = row
    expected_grid = {(trajectory, slot):(group[slot].get('eligible', True) if slot in group else False)
                     for trajectory, group in development_groups.items() for slot in range(56)}
    if not by_replicate or any(set(population) != set(expected_grid) or any(
            population[key].get('eligible') != eligible for key, eligible in expected_grid.items())
            for population in by_replicate.values()):
        raise ValueError('TMS development traces must preserve the complete manifest and fixed-slot eligibility grid')
    from .selection import score_development
    score_development([{'trajectory_id':trajectory, 'replicate':replicate,
                        'records':[population[(trajectory, slot)] for slot in range(56)]}
                       for replicate, population in by_replicate.items() for trajectory in development_groups])
    invoked = [row for row in rows if row.get('eligible') and row.get('invoked')]
    if not invoked:
        raise ValueError('TMS requires nonempty invoked H development traces')
    budgets = [float(value) for row in invoked for value in row['executed_budgets']]
    counts = [row['candidate_count'] for row in invoked]
    schedule = build_tms_schedule(task=args.task, family=args.family_id, population=args.population,
        population_manifest_sha256=file_hash(args.manifest), public_hash_seed=args.public_hash_seed,
        slots=[{'slot_id':slot_identifier(t,s),'original_step':s+1} for t, group in groups.items()
               for s,row in group.items() if row.get('eligible',True)],
        mean_regional_budget=float(np.mean(budgets, dtype=np.float64)), mean_candidate_count=float(np.mean(counts)),
        selected_h_checkpoint_sha256=file_hash(args.selected_h_checkpoint),
        development_manifest_sha256=file_hash(args.development_manifest))
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, schedule.to_dict())


def download_dataset_command(args):
    from .dataset_sources import download_dataset
    report = download_dataset(args.dataset, args.revision, args.output, include=args.include)
    print(json.dumps(report, indent=2))


def extract_images_command(args):
    from .dataset_sources import extract_images
    print(json.dumps(extract_images(args.archive, args.output), indent=2))


def convert_screenspot_command(args):
    from .dataset_sources import convert_screenspot
    print(json.dumps(convert_screenspot(args.annotations, args.images, args.output), indent=2))


def convert_gui360_command(args):
    from .dataset_sources import convert_gui360
    print(json.dumps(convert_gui360(args.root, args.output, task=args.task, role=args.role,
                                   action_schema=args.action_schema, action_map=args.action_map), indent=2))


def _training_input_bindings(args):
    from .evaluation import artifact_binding, file_hash
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    bindings = {'kind': 'new-reference-training-input-preparation',
                'runtime': versions(device=args.device, dtype=args.dtype)}
    for field in ('model', 'tokenizer', 'dino_model'):
        revision = 'revision' if field == 'model' else field + '_revision'
        bindings[field] = artifact_binding(getattr(args, field, None), getattr(args, revision, None))
    encoder = None
    if args.dino_model:
        if not args.public_projection:
            raise ValueError('--dino-model requires the saved --public-projection')
        from .features import DinoRegionalEncoder
        encoder = DinoRegionalEncoder.from_pretrained(args.dino_model, args.dino_revision,
            args.public_projection, args.device, local_files_only=not args.allow_download)
        bindings['public_projection_sha256'] = file_hash(args.public_projection)
    elif args.public_projection:
        raise ValueError('--public-projection requires --dino-model')
    return bindings, encoder


def prepare_alignment_command(args):
    from .training_data import QwenNativeTargetEncoder, build_alignment_data
    bindings, encoder = _training_input_bindings(args)
    native = None
    if args.model:
        native = QwenNativeTargetEncoder.from_pretrained(args.model, args.revision, device=args.device,
            dtype=args.dtype, local_files_only=not args.allow_download)
    report = build_alignment_data(args.manifest, args.output, feature_encoder=encoder,
                                  native_encoder=native, provenance=bindings)
    print(json.dumps({'output': args.output, 'eligible_records': report['eligible_records'], 'stage': 1}))


def prepare_supervised_command(args):
    from .training_data import load_training_tokenizer, build_supervised_data
    bindings, encoder = _training_input_bindings(args)
    tokenizer = load_training_tokenizer(args.tokenizer, args.tokenizer_revision,
                                       local_files_only=not args.allow_download)
    report = build_supervised_data(args.manifest, args.output, task=args.task, tokenizer=tokenizer,
        eos_token_id=args.eos_token_id, feature_encoder=encoder, provenance=bindings, action_schema=args.action_schema)
    print(json.dumps({'output': args.output, 'eligible_records': report['eligible_records'], 'stage': 2}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download-dataset', allow_abbrev=False, help='Download explicitly selected official dataset files at a pinned revision')
    download.add_argument('--dataset', choices=['screenspot-v2','gui360'], required=True)
    download.add_argument('--revision', required=True); download.add_argument('--output', required=True)
    download.add_argument('--include', action='append', help='Repeat for each file pattern; required for GUI-360')
    download.set_defaults(func=download_dataset_command)
    extract = commands.add_parser('extract-images', allow_abbrev=False, help='Validate and extract an image ZIP into a new directory')
    extract.add_argument('--archive', required=True); extract.add_argument('--output', required=True)
    extract.set_defaults(func=extract_images_command)
    screenspot = commands.add_parser('convert-screenspot', allow_abbrev=False, help='Convert original ScreenSpot-v2 annotations to evaluation records')
    screenspot.add_argument('--annotations', nargs='+', required=True)
    screenspot.add_argument('--images', required=True); screenspot.add_argument('--output', required=True)
    screenspot.set_defaults(func=convert_screenspot_command)
    gui360 = commands.add_parser('convert-gui360', allow_abbrev=False, help='Convert GUI-360 raw trajectories using explicit role and Action mappings')
    gui360.add_argument('--root', required=True); gui360.add_argument('--output', required=True)
    gui360.add_argument('--task', choices=['G','A'], required=True)
    gui360.add_argument('--role', choices=['train','dev','test'], required=True)
    gui360.add_argument('--action-schema'); gui360.add_argument('--action-map')
    gui360.set_defaults(func=convert_gui360_command)
    for command, handler in [('prepare-alignment', prepare_alignment_command), ('prepare-supervised', prepare_supervised_command)]:
        training = commands.add_parser(command, allow_abbrev=False, help='Build trusted fit-train inputs with explicit model bindings')
        training.add_argument('--manifest', required=True); training.add_argument('--output', required=True)
        training.add_argument('--dino-model'); training.add_argument('--dino-revision')
        training.add_argument('--public-projection'); training.add_argument('--device', default='cpu')
        training.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
        training.add_argument('--allow-download', action='store_true')
        if command == 'prepare-alignment':
            training.add_argument('--model'); training.add_argument('--revision')
        else:
            training.add_argument('--tokenizer', required=True); training.add_argument('--tokenizer-revision')
            training.add_argument('--task', choices=['G','A'], required=True)
            training.add_argument('--eos-token-id', type=int); training.add_argument('--action-schema')
        training.set_defaults(func=handler)
    prepare = commands.add_parser('prepare-dataset', allow_abbrev=False, help='Validate supplied records and explicit trajectory splits')
    for field in ('records','splits','output'): prepare.add_argument('--'+field, required=True)
    prepare.add_argument('--task', choices=['G','A'], required=True)
    prepare.add_argument('--action-schema')
    prepare.set_defaults(func=prepare_dataset_command)
    summary = commands.add_parser('summarize-run', allow_abbrev=False, help='Validate a completed run and export aggregate metrics')
    summary.add_argument('--run-directory', required=True); summary.add_argument('--output', required=True)
    summary.set_defaults(func=summarize_run_command)
    train = commands.add_parser('train', allow_abbrev=False, help='Fit with held-out development checkpoint selection')
    train.add_argument('--config'); train.add_argument('--replay', required=True)
    train.add_argument('--method', choices=METHODS, default='H')
    train.add_argument('--updates', type=int, required=True); train.add_argument('--batch-size', type=int, default=256)
    train.add_argument('--device', default='cpu'); train.add_argument('--seed', type=int, default=20260916)
    train.add_argument('--resume'); train.add_argument('--run-id'); train.add_argument('--output', required=True)
    train.add_argument('--dev-manifest'); train.add_argument('--dev-replicates', type=int, default=3)
    train.add_argument('--task', choices=['G','A'], default='G'); train.add_argument('--family-id', required=True)
    train.add_argument('--tms-schedule'); train.add_argument('--dev-tms-schedule')
    train.add_argument('--software-only', action='store_true', help='Synthetic fixture only; no selected model or benchmark claim')
    add_executor_arguments(train); train.set_defaults(func=fit)
    check = commands.add_parser('smoke', allow_abbrev=False, help='Synthetic CPU checks; not a benchmark experiment')
    check.add_argument('--config'); check.add_argument('--output', required=True)
    check.add_argument('--updates', type=int, default=3); check.set_defaults(func=smoke)
    for name, task in [('collect-grounding','G'),('collect-action','A')]:
        collection = commands.add_parser(name, allow_abbrev=False, help='Execute the frozen released-input recorded evaluation')
        collection.add_argument('--manifest', required=True); collection.add_argument('--output', required=True)
        collection.add_argument('--task', choices=[task], default=task)
        collection.add_argument('--device', default='cpu'); collection.add_argument('--seed', type=int, default=20260916)
        collection.add_argument('--family-id', required=True); collection.add_argument('--replicate-id', default='1')
        collection.add_argument('--split', choices=['fit-train','development','evaluation','test'])
        collection.add_argument('--controller-checkpoint'); collection.add_argument('--controller-config')
        collection.add_argument('--training-replay'); collection.add_argument('--controller-method', choices=METHODS, default='H')
        collection.add_argument('--tms-schedule'); collection.add_argument('--evaluation-tms-schedule')
        add_executor_arguments(collection); collection.set_defaults(func=collect)
    align = commands.add_parser('fit-projection', allow_abbrev=False, help='Fit Stage 1 against saved native visual tokens')
    align.add_argument('--records', required=True); align.add_argument('--output', required=True)
    align.add_argument('--device', default='cpu'); align.add_argument('--seed', type=int, default=20260916)
    align.set_defaults(func=fit_projection)
    adapter = commands.add_parser('fit-adapter', allow_abbrev=False, help='Stage 2 language q/v LoRA and projection teacher forcing')
    for field in ('model','projection','records','output'): adapter.add_argument('--'+field, required=True)
    adapter.add_argument('--revision'); adapter.add_argument('--device', default='cpu')
    adapter.add_argument('--seed', type=int, default=20260916); adapter.add_argument('--dtype', choices=['float32','bfloat16'], default='float32')
    adapter.add_argument('--allow-download', action='store_true'); adapter.add_argument('--save-merged', action='store_true')
    adapter.set_defaults(func=fit_adapter)
    tms = commands.add_parser('build-tms', allow_abbrev=False, help='Bind H development means to a public target-population schedule')
    for field in ('manifest','development-transcript','development-run','development-manifest','selected-h-checkpoint','family-id','public-hash-seed','population','output'):
        tms.add_argument('--'+field, required=True)
    tms.add_argument('--task', choices=['G','A'], required=True); tms.set_defaults(func=build_tms)
    tms.add_argument('--selection-state', help='Defaults to selection.json beside the selected H checkpoint')
    protocol = commands.add_parser('protocol', allow_abbrev=False, help='Validate and plan explicit 10-family/3-repeat runs; does not execute by default')
    protocol.add_argument('--registry', required=True); protocol.add_argument('--output-root', required=True)
    protocol.add_argument('--plan-output'); protocol.add_argument('--execute', action='store_true')
    protocol.set_defaults(func=protocol_command)
    args = parser.parse_args(argv)
    if hasattr(args, 'updates') and args.updates < 1: parser.error('--updates must be positive')
    if hasattr(args, 'dev_replicates') and args.dev_replicates < 1: parser.error('--dev-replicates must be positive')
    args.func(args)


if __name__=='__main__':main()
