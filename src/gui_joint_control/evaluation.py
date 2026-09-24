"""Recorded Grounding/Action driver shared by development and final evaluation.

This module wires supplied datasets and fitted artifacts to the mechanism. It
does not infer historical splits, recover measurements, or launch live GUIs.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib
import inspect
import io
import json
import math
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_binding(path, revision=None):
    """Bind local snapshot bytes or the explicitly pinned remote commit."""
    if not path:
        return None
    location = Path(path)
    if not location.is_dir():
        return {'identifier': str(path), 'revision': revision}
    from .protocol import artifact_digest
    return {'identifier': str(location.resolve()), 'tree_sha256': artifact_digest(location)}


def slot_identifier(trajectory_id, slot):
    """Stable unambiguous public key, also used in trusted replay metadata."""
    return json.dumps([str(trajectory_id), int(slot)], ensure_ascii=False, separators=(',', ':'))


def manifest_rows(path, task='G'):
    if task not in ('G', 'A'):
        raise ValueError('Expected Grounding (G) or Action (A)')
    groups = {}
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        trajectory, slot = str(row['trajectory_id']), row['slot']
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < 56:
            raise ValueError(f'Manifest line {number}: slot must be integer 0..55')
        if row.get('task', 'G') != task:
            raise ValueError(f'Manifest task differs from {"Grounding" if task == "G" else "Action"}')
        if not isinstance(row.get('eligible', True), bool):
            raise ValueError('eligible must be a boolean fixed by the manifest')
        group = groups.setdefault(trajectory, {})
        if slot in group:
            raise ValueError('Duplicate trajectory/slot key')
        group[slot] = row
    if not groups:
        raise ValueError('Empty manifest')
    return groups


def assert_disjoint_manifests(training_replay, development_manifest, *, task):
    """When collection provenance is available, reject development-ID leakage."""
    identifiers = training_replay.metadata.get('slot_id')
    if identifiers is None:
        raise ValueError('Development selection requires replay slot_id provenance')
    groups = manifest_rows(development_manifest, task)
    try:
        decoded = [json.loads(str(x)) for x in identifiers]
        if any(not isinstance(x, list) or len(x) != 2 or not isinstance(x[0], str) or type(x[1]) is not int for x in decoded):
            raise ValueError('Invalid public replay slot ID')
    except (ValueError, TypeError) as error:
        raise ValueError('Replay slot IDs must use the collector canonical [trajectory,slot] encoding') from error
    if set(groups).intersection(x[0] for x in decoded):
        raise ValueError('Training and development must not share trajectories, even at different slots')
    source_hashes = training_replay.metadata.get('source_manifest_sha256', [])
    if file_hash(development_manifest) in source_hashes:
        raise ValueError('Training and development cannot use the same manifest')


def decoder_seed(seed, trajectory, slot, candidate, *, family='', task='G', replicate='1'):
    """Paired methods omit the method label and preserve nested candidate prefixes."""
    fields = ('JDC-decoder-v2', str(seed), str(family), task, str(replicate),
              str(trajectory), str(slot), str(candidate))
    encoded = b''.join(len(x.encode('utf-8')).to_bytes(8, 'big') + x.encode('utf-8') for x in fields)
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**63 - 1)


def validate_tms_population(schedule, manifest, *, task, family):
    groups = manifest_rows(manifest, task)
    if (schedule.population_manifest_sha256 != file_hash(manifest)
            or schedule.task != task or schedule.family != family):
        raise ValueError('TMS schedule is bound to a different population/task/family')
    expected = {slot_identifier(t, s) for t, group in groups.items() for s, row in group.items() if row.get('eligible', True)}
    if {row['slot_id'] for row in schedule.to_dict()['slots']} != expected:
        raise ValueError('TMS schedule must bind every eligible population ID exactly')


def render_prompt(payload, *, action_schema_text=''):
    """Retrieved typed JSON precedes the verbatim current instruction."""
    if payload.task == 'G':
        wrapper = 'Return only a JSON object with normalized coordinates: {"x": number, "y": number}.\n'
    else:
        wrapper = 'Return exactly one JSON object with function, arguments, and status.\n'
        wrapper += 'Frozen action schema: ' + action_schema_text + '\n'
    demos = ''
    if payload.retrieval:
        # Each item was validated and canonicalized by FrozenDemonstrationBank.
        demos = 'Demonstrations: [' + ','.join(x.text for x in payload.retrieval) + ']\n'
    text = wrapper + demos + 'Instruction: ' + payload.current
    if payload.history:
        text += '\nPrevious reference thoughts:\n' + '\n'.join(payload.history)
    return text


def load_evaluator(spec):
    """Load an explicitly requested local evaluator; never guess official rules."""
    if not spec or ':' not in spec:
        raise ValueError('Action requires --action-evaluator module:function for the pinned evaluator')
    module, name = spec.split(':', 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise ValueError('Action evaluator must be callable')
    source = inspect.getsourcefile(fn)
    if not source or not Path(source).is_file():
        raise ValueError('Action evaluator must have a hashable source file')
    return fn, {'entry_point': spec, 'source_sha256': file_hash(source)}


class RecordedEvaluator:
    """One frozen executor reused for every checkpoint or post-fit replicate."""

    def __init__(self, args, *, model=None, encoder=None, action_evaluator=None):
        from .executor import ReleasedQwenExecutor
        from .features import DinoRegionalEncoder
        self.args = args
        self.task = args.task
        self.model = model
        if self.model is None:
            if not args.model or not args.projection:
                raise ValueError('Evaluation requires an explicitly fitted --model and --projection')
            self.model = ReleasedQwenExecutor.from_pretrained(
                args.model, revision=args.revision, tokenizer_id=args.tokenizer or args.model,
                tokenizer_revision=args.tokenizer_revision or args.revision,
                projection=np.load(args.projection, allow_pickle=False), device=args.device,
                dtype=args.dtype, local_files_only=not args.allow_download)
        self.encoder = encoder
        self.bank = None
        self.schemas = None
        self.action_evaluator = action_evaluator
        self.evaluator_binding = None
        self.action_schema_text = ''
        self._trusted_input_hashes = {}
        if self.task == 'A':
            from .action_evaluation import load_action_schemas
            if not args.action_schema:
                raise ValueError('Action requires an explicit frozen --action-schema')
            self.schemas = load_action_schemas(args.action_schema)
            self.action_schema_text = Path(args.action_schema).read_text(encoding='utf-8')
            if self.action_evaluator is None:
                self.action_evaluator, self.evaluator_binding = load_evaluator(args.action_evaluator)
        if args.retrieval_bank or args.retrieval_keys:
            from .retrieval import FrozenDemonstrationBank
            if not args.retrieval_bank or not args.retrieval_keys or args.retrieval_threshold is None:
                raise ValueError('Retrieval requires bank, keys and a development-selected threshold')
            self.bank = FrozenDemonstrationBank.from_files(
                args.retrieval_bank, args.retrieval_keys, task=self.task, view=args.retrieval_view, schemas=self.schemas)
        elif not args.disable_retrieval:
            raise ValueError('Supply the frozen retrieval bank or explicitly request --disable-retrieval')
        if args.retrieval_threshold is not None and (
                not math.isfinite(args.retrieval_threshold) or args.retrieval_threshold not in np.arange(0, 2.01, .25)):
            raise ValueError('Retrieval threshold must belong to the development grid 0,.25,...,2')
        self.provenance = {
            'executor': self.model.provenance,
            'model_snapshot': artifact_binding(args.model, args.revision),
            'tokenizer_snapshot': artifact_binding(args.tokenizer or args.model, args.tokenizer_revision or args.revision),
            'dino_snapshot': artifact_binding(args.dino_model, args.dino_revision),
            'public_projection_sha256': file_hash(args.public_projection) if args.public_projection else None,
            'projection_sha256': file_hash(args.projection) if args.projection else None,
            'action_schema_sha256': file_hash(args.action_schema) if args.action_schema else None,
            'action_evaluator': self.evaluator_binding,
            'retrieval_bank_sha256': file_hash(args.retrieval_bank) if args.retrieval_bank else None,
            'retrieval_keys_sha256': file_hash(args.retrieval_keys) if args.retrieval_keys else None,
            'retrieval_view': args.retrieval_view if self.bank else None,
            'retrieval_threshold': args.retrieval_threshold if self.bank else None,
            'retrieval_enabled': self.bank is not None,
        }

    def run(self, manifest, *, trainer=None, replicate='1', output=None, tms_schedule=None):
        from .runtime import (Slot, Prediction, behavior_allocator, run_trajectory,
                              run_action_trajectory, save_replay)
        from .action_evaluation import ActionSlot, ActionPrediction
        from .prompt_policy import prepare_prompt
        from .features import DinoRegionalEncoder
        import torch
        groups = manifest_rows(manifest, self.task)
        base = Path(manifest).resolve().parent
        manifest_sha256 = file_hash(manifest)
        if trainer is not None and trainer.method in ('Disclosure-only', 'Count-only') and tms_schedule is None:
            raise ValueError('Single-head evaluation requires an explicit evaluation-population TMS schedule')
        if self.bank is not None:
            exclusion = getattr(self.args, 'retrieval_exclusion_manifest', None) or manifest
            self.bank.validate_evaluation_manifest(exclusion)
            if Path(exclusion).resolve() != Path(manifest).resolve():
                exclusion_groups = manifest_rows(exclusion, self.task)
                for trajectory, group in groups.items():
                    for slot, row in group.items():
                        bound = exclusion_groups.get(trajectory, {}).get(slot)
                        if bound is None or bound.get('public_metadata') != row.get('public_metadata'):
                            raise ValueError('Current evaluation rows must belong to the complete frozen exclusion manifest')
        if tms_schedule is not None:
            validate_tms_population(tms_schedule, manifest, task=self.task, family=self.args.family_id)
        # Validate public mandatory text and offline reference schema before any
        # screen access. Full prompts are rechecked after release-only retrieval.
        for group in groups.values():
            for row in group.values():
                if not row.get('eligible', True):
                    continue
                current = row.get('instruction', '') if self.task == 'G' else row.get('request', row.get('instruction', ''))
                prepared = prepare_prompt(self.task, current, (), (), lambda payload:
                    self.model.prompt_token_ids(render_prompt(payload, action_schema_text=self.action_schema_text)).reshape(-1).tolist())
                self.model.instruction_embeddings(prepared.payload.scoring_text)
                if self.task == 'A':
                    from .scoring import parse_action
                    reference = row.get('reference_action')
                    if reference is None or parse_action(json.dumps(reference, ensure_ascii=False, allow_nan=False), self.schemas) != reference:
                        raise ValueError('Eligible Action references must already be canonical under the frozen schema')
        if self.encoder is None and any('image_path' in r and 'features_path' not in r
                                       for group in groups.values() for r in group.values()):
            if not self.args.dino_model or not self.args.public_projection:
                raise ValueError('Image manifests require --dino-model and --public-projection')
            self.encoder = DinoRegionalEncoder.from_pretrained(
                self.args.dino_model, self.args.dino_revision, self.args.public_projection, self.args.device,
                local_files_only=not self.args.allow_download)
        if output is not None:
            output = Path(output)
            (output / 'releases').mkdir(parents=True, exist_ok=False)
        episodes, all_transitions, rows_out = [], [], []
        behavior = behavior_allocator(np.random.default_rng(self.args.seed))
        for trajectory, group in sorted(groups.items()):
            active = {}
            slots = []
            for index in range(max(group) + 1):
                row = group.get(index)
                if self.task == 'G':
                    slots.append(Slot('', None, False, False) if row is None else Slot(
                        row.get('instruction', ''), tuple(row['target_box']) if row.get('target_box') is not None else None,
                        row.get('eligible', True), True))
                else:
                    history = tuple(row.get('history', ())) if row else ()
                    slots.append(ActionSlot('', None, (), False, False) if row is None else ActionSlot(
                        row.get('request', row.get('instruction', '')), row.get('reference_action'), history,
                        row.get('eligible', True), True,
                        tuple(row['screen_size']) if row.get('screen_size') else None,
                        row.get('reference_boxes')))

            def load_features(index):
                active['slot'] = index
                row = group[index]
                field = 'features_path' if 'features_path' in row else 'image_path'
                if field not in row:
                    raise ValueError('Every eligible row requires features_path or image_path')
                source = (base / row[field]).resolve()
                # Private bytes are read and hashed only AFTER public admission.
                # Their hashes remain a separate trusted-local audit artifact.
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                previous = self._trusted_input_hashes.setdefault(str(source), digest)
                if previous != digest:
                    raise ValueError('A bound private feature/image input changed during evaluation')
                if field == 'features_path':
                    return np.load(io.BytesIO(data), allow_pickle=False)
                return self.encoder(io.BytesIO(data))

            def allocate(observation):
                if trainer is None:
                    if tms_schedule is None:
                        return behavior(observation)
                    budgets, counts = tms_schedule.proposals([slot_identifier(trajectory, active['slot'])])
                    return budgets[0], int(counts[0])
                state = torch.tensor(np.asarray(observation)[None], dtype=torch.float32, device=trainer.device)
                with torch.inference_mode():
                    action = trainer.proposal_action(state, slot_ids=[slot_identifier(trajectory, active['slot'])],
                                                     tms_schedule=tms_schedule)
                return action['budgets'][0].cpu().numpy(), int(action['candidate_count'][0])

            def execute(release, text, history, count, prior_feedback):
                retrieved = () if self.bank is None else self.bank.retrieve(
                    release, prior_feedback, self.args.retrieval_threshold)
                render = lambda payload: render_prompt(payload, action_schema_text=self.action_schema_text)
                prepared = prepare_prompt(self.task, text, history, retrieved,
                    lambda payload: self.model.prompt_token_ids(render(payload)).reshape(-1).tolist())
                seeds = [decoder_seed(self.args.seed, trajectory, active['slot'], i,
                         family=self.args.family_id, task=self.task, replicate=replicate) for i in range(count)]
                kwargs = dict(prompt_text=render(prepared.payload), scoring_text=prepared.payload.scoring_text, seeds=seeds)
                if self.task == 'G':
                    result, candidates = self.model.predict_grounding(release, **kwargs)
                    prediction = Prediction(result.coordinate, result.feedback)
                else:
                    action, feedback, candidates = self.model.predict_action(release, schemas=self.schemas, **kwargs)
                    result = None
                    prediction = ActionPrediction(action, feedback)
                if output is not None:
                    name = hashlib.sha256(slot_identifier(trajectory, active['slot']).encode('utf-8')).hexdigest() + '.npy'
                    np.save(output / 'releases' / name, release, allow_pickle=False)
                    exported = {'trajectory_id': trajectory, 'slot': active['slot'], 'task': self.task,
                        'family_id': self.args.family_id, 'replicate_id': str(replicate),
                        'release_file': 'releases/' + name, 'public_decoder_seeds': seeds,
                        'input_ids': prepared.input_ids, 'scoring_text': prepared.payload.scoring_text,
                        'scoring_token_ids': self.model.tokenizer.encode(prepared.payload.scoring_text, add_special_tokens=False),
                        'retrieval_count': len(prepared.payload.retrieval),
                        'removed_history': prepared.removed_history, 'removed_retrieval': prepared.removed_retrieval,
                        'candidates': [asdict(c) for c in candidates]}
                    if hasattr(result, 'selected_index'):
                        exported['selected_index'] = result.selected_index
                    with (output / 'candidates.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(exported, ensure_ascii=False, allow_nan=False) + '\n')
                return prediction

            if self.task == 'G':
                records, transitions = run_trajectory(slots, load_features, allocate, execute, executor_context=True)
            else:
                records, transitions = run_action_trajectory(
                    slots, load_features, allocate, execute, evaluator=self.action_evaluator)
            invoked_ids = [slot_identifier(trajectory, r['slot']) for r in records if r['invoked']]
            for index, transition in enumerate(transitions):
                transition['slot_id'] = invoked_ids[index]
                transition['next_slot_id'] = invoked_ids[index + 1] if index + 1 < len(invoked_ids) else ''
                transition['source_manifest_sha256'] = manifest_sha256
                transition['task'] = self.task
                transition['family_id'] = self.args.family_id
            all_transitions.extend(transitions)
            episodes.append({'trajectory_id': trajectory, 'replicate': str(replicate), 'records': records})
            rows_out.extend({'trajectory_id': trajectory, **record} for record in records)
        if output is not None:
            with (output / 'transcript.jsonl').open('w', encoding='utf-8') as stream:
                for row in rows_out:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            if trainer is None and tms_schedule is None:
                save_replay(all_transitions, output / 'replay.npz')
            (output / 'trusted-inputs.json').write_text(json.dumps({
                'scope':'trusted local only; not protected public mechanism output',
                'admitted_input_sha256':self._trusted_input_hashes}, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        eligible = sum(r['eligible'] for r in rows_out)
        invoked = sum(r['invoked'] for r in rows_out)
        correct = sum(r['correct'] for r in rows_out)
        report = {'task': self.task, 'family_id': self.args.family_id, 'replicate_id': str(replicate),
                  'eligible': eligible, 'invoked': invoked, 'correct': correct,
                  'accuracy': correct / eligible if eligible else None,
                  'manifest_sha256': file_hash(manifest), 'public_seed': self.args.seed,
                  'decoder_seed_scheme': 'JDC-decoder-v2: seed/family/task/replicate/trajectory/slot/candidate',
                  'model_dtype': self.args.dtype, **self.provenance, 'reproduces_historical_results': False}
        if self.task == 'A':
            for field in ('function_correct', 'arguments_correct', 'status_correct'):
                report[field] = sum(bool(r.get(field, False)) for r in rows_out)
                report[field.replace('_correct', '_accuracy')] = report[field] / eligible if eligible else None
        return report, episodes
