"""Pinned upstream downloads and explicit raw-benchmark interchange adapters."""
from fnmatch import fnmatchcase
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from PIL import Image

from .dataset_preparation import _box, _json


DATASETS = {'screenspot-v2': 'OS-Copilot/ScreenSpot-v2', 'gui360': 'vyokky/GUI-360'}
SCREENSPOT_FILES = [f'screenspot_{p}_v2.json' for p in ('desktop', 'mobile', 'web')]


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def download_dataset(dataset, revision, output_dir, *, include=None):
    """Download an explicit immutable snapshot; never auto-extract or run code."""
    if dataset not in DATASETS or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Select a supported dataset and immutable 40-hex revision')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    patterns = list(include or (SCREENSPOT_FILES + ['screenspotv2_image.zip'] if dataset == 'screenspot-v2' else []))
    if not patterns or any(not isinstance(p, str) or not p.strip() for p in patterns):
        raise ValueError('GUI-360 requires explicit --include file patterns')
    from huggingface_hub import HfApi, snapshot_download
    names = HfApi().list_repo_files(DATASETS[dataset], repo_type='dataset', revision=revision)
    for pattern in patterns:
        if not any(fnmatchcase(name, pattern) for name in names):
            raise ValueError(f'Include pattern matches no upstream files: {pattern}')
    selected = sorted(name for name in names if any(fnmatchcase(name, p) for p in patterns))
    snapshot_download(DATASETS[dataset], repo_type='dataset', revision=revision,
                      allow_patterns=selected, local_dir=str(output), max_workers=4)
    if any(not (output / name).is_file() for name in selected):
        raise ValueError('Incomplete download; selected upstream files are missing')
    report = {'dataset': dataset, 'repo_id': DATASETS[dataset], 'revision': revision,
              'files': selected, 'include': patterns, 'scope': 'upstream snapshot; not a manuscript split'}
    _write(output / 'download.json', report)
    return report


def extract_images(archive_path, output_dir):
    """Extract a ZIP into a fresh directory after validating every member path."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive_path) as archive:
        names = set()
        for member in archive.infolist():
            name = member.filename
            parts = PurePosixPath(name).parts
            if (not name or '\\' in name or name.startswith('/') or '..' in parts
                    or any(re.search(r'[\x00-\x1f<>:"|?*]', part) or part.rstrip(' .') != part
                           or re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part)
                           for part in parts)
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe ZIP member: {name}')
            key = PurePosixPath(name).as_posix().rstrip('/').casefold()
            if key in names:
                raise ValueError(f'Duplicate ZIP member: {name}')
            names.add(key)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.extract-') as temporary:
            stage = Path(temporary) / 'images'
            stage.mkdir()
            archive.extractall(stage)
            stage.rename(output)
    return {'archive': str(Path(archive_path).resolve()), 'output': str(output), 'members': len(names)}


def _asset(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Expected a relative POSIX image path')
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Image path escapes its declared root')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping image: {relative}')
    return path


def _image_size(path):
    with Image.open(path) as image:
        return list(image.size)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _finish(output, rows, excluded, sources, *, dataset, role, task, rules):
    if output.exists():
        raise FileExistsError(output)
    if not rows and not excluded:
        raise ValueError('No retained records in this conversion')
    keys = [(row['trajectory_id'], row['slot']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate trajectory/slot identities')
    assignments = {name: sorted({r['trajectory_id'] for r in rows}) if name == role else []
                   for name in ('train', 'dev', 'test')}
    report = {'dataset': dataset, 'task': task, 'assigned_role': role, 'records': len(rows),
              'eligible_records': sum(row['eligible'] for row in rows), 'exclusions': len(excluded),
              'sources': sources, 'rules': rules, 'reproduces_historical_results': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.convert-') as temporary:
        stage = Path(temporary) / 'converted'
        stage.mkdir()
        for name, values in [('records.jsonl', rows), ('exclusions.jsonl', excluded)]:
            (stage / name).write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                             for row in values), encoding='utf-8')
        _write(stage / 'splits.json', assignments)
        _write(stage / 'conversion.json', report)
        stage.rename(output)
    return report


def convert_screenspot(annotation_paths, images_dir, output_dir):
    """Preserve every annotation row (including duplicates) as one test episode."""
    output, images = Path(output_dir).resolve(), Path(images_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows, sources = [], []
    for value in annotation_paths:
        path = Path(value).resolve()
        raw = path.read_bytes()
        annotations = _json(raw.decode('utf-8-sig'))
        if not isinstance(annotations, list):
            raise ValueError('ScreenSpot annotations must be a JSON array')
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for index, item in enumerate(annotations):
            image = _asset(images, item['img_filename'])
            width, height = _image_size(image)
            bbox = item['bbox']
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)
                    or bbox[2] < 0 or bbox[3] < 0):
                raise ValueError('ScreenSpot bbox must be pixel [x,y,width,height]')
            x, y, w, h = bbox
            rows.append({'trajectory_id': f'screenspot-v2/{path.stem}/{index:06d}', 'slot': 0,
                         'task': 'G', 'split': 'test', 'eligible': True,
                         'instruction': _text(item['instruction'], 'instruction'),
                         'target_box': _box([x, y, x+w, y+h], [width, height]),
                         'image_path': Path(os.path.relpath(image, output)).as_posix(),
                         'screen_size': [width, height], 'upstream_metadata': {
                             'image': item['img_filename'], 'data_type': item.get('data_type'),
                             'data_source': item.get('data_source'), 'annotation_row': index}})
    return _finish(output, rows, [], sources, dataset='screenspot-v2', role='test', task='G',
                   rules={'bbox': 'pixel xywh -> normalized xyxy', 'duplicates': 'retained', 'split': 'evaluation only'})


def _mapped_action(action, status, size, schemas, mappings):
    from .scoring import parse_action
    function = action.get('function')
    if function not in schemas:
        return None, {}, []
    mapping = mappings.get(function)
    if not isinstance(mapping, dict) or not isinstance(mapping.get('arguments'), dict):
        raise ValueError(f'Explicit Action argument mapping required for {function}')
    def source_value(key):
        if key.startswith('action.'):
            return action[key[len('action.'):]]
        return action['args'][key.removeprefix('args.')]
    args = {}
    for name, fields in mapping['arguments'].items():
        if not (isinstance(fields, str) or isinstance(fields, list) and len(fields) == 2 and all(isinstance(f, str) for f in fields)):
            raise ValueError('Argument mapping needs a source key or a pair of source keys')
        try:
            value = source_value(fields) if isinstance(fields, str) else [source_value(f) for f in fields]
        except KeyError:
            if name in schemas[function].optional and isinstance(fields, str):
                continue
            raise ValueError(f'Missing source field for mapped argument {name}') from None
        if name in schemas[function].spatial:
            if (not isinstance(value, list) or len(value) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
                raise ValueError('Spatial Action source arguments must be finite pixel pairs')
            value = [value[0] / size[0], value[1] / size[1]]
        args[name] = value
    # This is the upstream evaluator's explicit status mapping.
    status = {'OVERALL_FINISH': 'FINISH', 'FINISH': 'CONTINUE'}.get(status, status)
    canonical = parse_action(json.dumps({'function': function, 'arguments': args, 'status': status}), schemas)
    if canonical is None:
        raise ValueError('Mapped Action target is invalid under the supplied frozen schema')
    boxes, unusable_boxes = {}, []
    for argument, field in mapping.get('boxes', {}).items():
        if argument not in schemas[function].spatial:
            raise ValueError('Reference box mapping must name a spatial argument')
        rect = action.get(field)
        if rect:
            try:
                values = [rect[k] for k in ('left', 'top', 'right', 'bottom')]
                if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                        or values[0] > values[2] or values[1] > values[3]):
                    raise ValueError('Unparseable reference rectangle')
                # A parseable off-screen rectangle still supports containment.
                boxes[argument] = [values[i] / size[i % 2] for i in range(4)]
            except (KeyError, TypeError, ValueError):
                # Keep the eligible target for the official point-distance fallback.
                unusable_boxes.append({'argument': argument, 'source_field': field})
    return canonical, boxes, unusable_boxes


def convert_gui360(root_dir, output_dir, *, task, role, action_schema=None, action_map=None):
    """Read official raw data/image layout; role assignment remains caller supplied."""
    from .action_evaluation import load_action_schemas
    if task not in ('G', 'A') or role not in ('train', 'dev', 'test'):
        raise ValueError('Explicit task G/A and role train/dev/test required')
    root, output = Path(root_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    schemas, mappings = None, None
    sources = []
    if task == 'A':
        if action_schema is None or action_map is None:
            raise ValueError('Action conversion requires --action-schema and --action-map')
        schemas = load_action_schemas(action_schema)
        mappings = _json(Path(action_map).read_text(encoding='utf-8'))
        if not isinstance(mappings, dict):
            raise ValueError('Action map must be an object keyed by upstream function')
        for path in (action_schema, action_map):
            sources.append({'path': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    rows, excluded = [], []
    for path in sorted((root / 'data').rglob('*.jsonl')):
        relative = path.relative_to(root / 'data')
        if len(relative.parts) != 4:
            raise ValueError('Expected data/<domain>/<category>/<success-or-fail>/<trajectory>.jsonl')
        raw = path.read_bytes()
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        items = [_json(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        if not items:
            raise ValueError(f'Empty trajectory: {relative}')
        ids = [r.get('step_id') for r in items]
        if any(type(i) is not int or i < 1 for i in ids) or ids != list(range(1, len(items)+1)):
            raise ValueError('Raw step_id must be complete, ordered and one-based')
        if any(r.get('total_steps') != len(items) for r in items):
            raise ValueError('Raw total_steps must match the complete trajectory')
        execution = _text(items[0]['execution_id'], 'execution_id')
        if any(r['execution_id'] != execution for r in items):
            raise ValueError('One JSONL file must represent one execution_id')
        trajectory = '/'.join(relative.parts[:2]) + '/' + execution
        reason = 'exceeds_56_slot_horizon' if len(items) > 56 else None
        if task == 'A':
            types = [r['step']['action'].get('action_type') for r in items]
            if any(t not in ('GUI', 'API') for t in types):
                raise ValueError('Every Action step requires explicit GUI/API action_type')
            if any(t == 'API' and r['step']['action'].get('function') for t, r in zip(types, items)):
                reason = reason or 'trajectory_contains_API_action'
        if reason:
            excluded.append({'trajectory_id': trajectory, 'reason': reason, 'records': len(items)})
            continue
        history = []
        for item in items:
            step, slot = item['step'], item['step_id']-1
            action = step['action']
            tag = 'grounding' if task == 'G' else 'action_prediction'
            if not isinstance(step['tags'], list):
                raise ValueError('Raw tags must be an array')
            eligible = tag in step['tags']
            if task == 'G':
                eligible = eligible and bool(action.get('rectangle'))
            else:
                eligible = eligible and action.get('function') in schemas
            row = {'trajectory_id': trajectory, 'slot': slot, 'task': task, 'split': role, 'eligible': eligible,
                   'upstream_metadata': {'execution_id': execution, 'step_id': item['step_id'],
                                         'raw_file': relative.as_posix(), 'task_tags': step['tags']}}
            if eligible:
                image_root = root / 'image' / relative.parts[0] / relative.parts[1]
                image = _asset(image_root, step['screenshot_clean'])
                size = _image_size(image)
                row.update(image_path=Path(os.path.relpath(image, output)).as_posix(), screen_size=size)
                if task == 'G':
                    rectangle = action['rectangle']
                    row.update(instruction=_text(step['thought'], 'thought'),
                               target_box=_box([rectangle[k] for k in ('left', 'top', 'right', 'bottom')], size))
                    point = [action.get('coordinate_x'), action.get('coordinate_y')]
                    if all(v is not None for v in point):
                        # Keep the explicit raw target; never substitute a box center.
                        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)
                                or not 0 <= point[0] <= size[0] or not 0 <= point[1] <= size[1]):
                            raise ValueError('Raw Grounding coordinate must be a finite in-screen pixel point')
                        row['target_text'] = json.dumps({'x': point[0]/size[0], 'y': point[1]/size[1]})
                else:
                    canonical, boxes, unusable_boxes = _mapped_action(action, step['status'], size, schemas, mappings)
                    row.update(request=_text(item['request'], 'request'), history=list(history),
                               reference_action=canonical, reference_boxes=boxes)
                    row['upstream_metadata']['unusable_reference_boxes'] = unusable_boxes
            else:
                row['exclusion_reason'] = 'task_tag_or_target_not_supported'
            rows.append(row)
            if task == 'A':
                history.append(_text(step['thought'], 'preceding thought'))
    return _finish(output, rows, excluded, sources, dataset='gui360', role=role, task=task,
                   rules={'slots': 'complete upstream step_id minus one; no renumbering',
                          'Action_API': 'exclude whole trajectory with a nonempty API function', 'Grounding_text': 'current thought',
                          'Action_text': 'request plus preceding raw thoughts', 'Action_boxes': 'normalized xyxy',
                          'split': 'caller-assigned; apply your train/dev assignment before prepare-dataset',
                          'scope': 'raw upstream conversion; not the paper processed-inventory joins'})
