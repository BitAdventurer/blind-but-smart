"""Small upstream-shaped fixtures; no network or benchmark experiments."""
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from gui_joint_control.dataset_sources import (
    convert_gui360, convert_screenspot, download_dataset, extract_images,
)
from gui_joint_control.dataset_preparation import prepare_dataset


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_screenspot_xywh_duplicates_and_prepare_round_trip(tmp_path):
    images = tmp_path/'images'; images.mkdir()
    Image.new('RGB', (200,100)).save(images/'screen.png')
    row = {'img_filename':'screen.png','instruction':'Click Save','bbox':[20,10,40,20],
           'data_type':'text','data_source':'windows'}
    annotation = tmp_path/'screenspot_desktop_v2.json'
    write_json(annotation, [row, row])
    output = tmp_path/'converted'
    convert_screenspot([annotation], images, output)
    rows = [json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[0]['trajectory_id'] != rows[1]['trajectory_id']
    assert rows[0]['target_box']==[.1,.1,.3,.3]
    assert all(r['split']=='test' and r['slot']==0 for r in rows)
    assert not any('target_text' in r for r in rows)
    report = prepare_dataset(output/'records.jsonl', output/'splits.json', tmp_path/'prepared')
    assert report['counts']['test']['eligible_records']==2
    with pytest.raises(FileExistsError):
        convert_screenspot([annotation], images, output)


@pytest.mark.parametrize('filename,bbox', [('../outside.png',[0,0,2,2]),
                                          ('screen.png',[190,0,20,5]),
                                          ('screen.png',[0,0,-1,5])])
def test_screenspot_rejects_unsafe_path_or_bad_geometry_without_output(tmp_path, filename, bbox):
    Image.new('RGB',(200,100)).save(tmp_path/'screen.png')
    annotation = tmp_path/'rows.json'
    write_json(annotation,[{'img_filename':filename,'bbox':bbox,'instruction':'Click'}])
    with pytest.raises(ValueError):
        convert_screenspot([annotation], tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def gui_fixture(tmp_path, count=2):
    root = tmp_path/'raw'
    image = root/'image/excel/in_app/success/example/screen.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB',(200,100)).save(image)
    rows = [{'execution_id':'example','step_id':i+1,'total_steps':count,'request':'Save workbook',
             'step':{'screenshot_clean':'success/example/screen.png','thought':f'thought {i}',
                     'status':'CONTINUE','tags':['grounding','action_prediction'],
                     'action':{'action_type':'GUI','function':'click','args':{'x':40,'y':20},
                               'rectangle':{'left':20,'top':10,'right':60,'bottom':30},
                               'coordinate_x':40,'coordinate_y':20}}} for i in range(count)]
    path = root/'data/excel/in_app/success/example.jsonl'
    save_trajectory(path, rows)
    return root, path, rows


def save_trajectory(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')


def schema_fixture(tmp_path):
    schema, mapping = tmp_path/'schema.json', tmp_path/'mapping.json'
    write_json(schema,{'schema_id':'fixture','source_split':'fit-train','source_manifest_sha256':'a'*64,
                       'functions':{'click':{'required':['point'],'optional':[],'spatial':['point'],
                                             'statuses':['CONTINUE','FINISH']}}})
    write_json(mapping, {'click':{'arguments':{'point':['x','y']},'boxes':{'point':'rectangle'}}})
    return schema, mapping


def test_gui_grounding_preserves_raw_slots_and_explicit_target(tmp_path):
    root, path, rows = gui_fixture(tmp_path)
    rows[0]['step']['tags'] = []
    save_trajectory(path, rows)
    output = tmp_path/'converted'
    convert_gui360(root, output, task='G', role='train')
    converted = [json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    assert [r['slot'] for r in converted]==[0,1]
    assert not converted[0]['eligible'] and converted[1]['eligible']
    assert converted[1]['instruction']=='thought 1'
    assert json.loads(converted[1]['target_text'])=={'x':.2,'y':.2}
    assert converted[1]['target_box']==[.1,.1,.3,.3]
    assert 'ui_tree' not in converted[1]
    report = prepare_dataset(output/'records.jsonl', output/'splits.json', tmp_path/'prepared')
    assert report['counts']['train']['accepted_records']==2


def test_gui_action_canonical_pixels_history_status_and_prepare(tmp_path):
    root, path, rows = gui_fixture(tmp_path)
    rows[0]['step']['tags']=[]
    rows[1]['step']['status']='OVERALL_FINISH'
    save_trajectory(path, rows)
    schema, mapping = schema_fixture(tmp_path)
    output = tmp_path/'converted'
    convert_gui360(root, output, task='A', role='test', action_schema=schema, action_map=mapping)
    records = [json.loads(line) for line in (output/'records.jsonl').read_text().splitlines()]
    assert records[1]['history']==['thought 0']
    assert records[1]['reference_action']=={'function':'click','arguments':{'point':[.2,.2]},'status':'FINISH'}
    assert records[1]['reference_boxes']=={'point':[.1,.1,.3,.3]}
    prepare_dataset(output/'records.jsonl', output/'splits.json', tmp_path/'prepared', task='A', action_schema=schema)


@pytest.mark.parametrize('reason', ['API','length'])
def test_gui_whole_trajectory_exclusions(tmp_path, reason):
    root, path, rows = gui_fixture(tmp_path, count=57 if reason=='length' else 2)
    schema, mapping = schema_fixture(tmp_path)
    if reason=='API':
        rows[1]['step']['action']['action_type']='API'
        rows[1]['step']['action']['function']='write_cell'
        save_trajectory(path, rows)
    report = convert_gui360(root, tmp_path/'out', task='A', role='test', action_schema=schema, action_map=mapping)
    assert report['records']==0 and report['exclusions']==1
    assert not (tmp_path/'out/records.jsonl').read_text()


def test_gui_rejects_partial_trajectory_and_missing_mapping(tmp_path):
    root, path, rows = gui_fixture(tmp_path)
    rows[1]['step_id']=3
    save_trajectory(path, rows)
    with pytest.raises(ValueError, match='step_id'):
        convert_gui360(root, tmp_path/'out', task='G', role='train')
    assert not (tmp_path/'out').exists()
    rows[1]['step_id']=2
    save_trajectory(path, rows)
    schema,mapping = schema_fixture(tmp_path)
    write_json(mapping,{})
    with pytest.raises(ValueError, match='mapping'):
        convert_gui360(root,tmp_path/'out',task='A',role='test',action_schema=schema,action_map=mapping)


@pytest.mark.parametrize('rect', [{'left': 50, 'top': 0, 'right': 40, 'bottom': 20}, {'left': 1}, [1,2]])
def test_action_optional_bad_box_preserves_eligible_fallback_target(tmp_path, rect):
    root, path, rows = gui_fixture(tmp_path)
    rows[0]['step']['action']['rectangle'] = rect
    save_trajectory(path, rows)
    schema, mapping = schema_fixture(tmp_path)
    report = convert_gui360(root, tmp_path/'out', task='A', role='test', action_schema=schema, action_map=mapping)
    record = json.loads((tmp_path/'out/records.jsonl').read_text().splitlines()[0])
    assert report['eligible_records']==2 and record['eligible']
    assert record['reference_boxes']=={}
    assert record['reference_action']['arguments']['point']==[.2,.2]
    assert record['upstream_metadata']['unusable_reference_boxes']


def test_action_maps_official_top_level_coordinates_and_keeps_parseable_boxes(tmp_path):
    root,path,rows=gui_fixture(tmp_path)
    for row in rows:
        row['step']['action']['args']={}
        row['step']['action']['rectangle']['left']=-2
    save_trajectory(path,rows)
    schema,mapping=schema_fixture(tmp_path)
    write_json(mapping, {'click': {'arguments': {'point': ['action.coordinate_x','action.coordinate_y']},
                                   'boxes': {'point':'rectangle'}}})
    convert_gui360(root,tmp_path/'out',task='A',role='test',action_schema=schema,action_map=mapping)
    record=json.loads((tmp_path/'out/records.jsonl').read_text().splitlines()[0])
    assert record['reference_action']['arguments']['point']==[.2,.2]
    assert record['reference_boxes']['point'][0]==-.01
    assert record['upstream_metadata']['unusable_reference_boxes']==[]


@pytest.mark.parametrize('names', [['images/a.png','images/./a.png'], ['a?.png'], ['NUL.png'], ['a.']])
def test_zip_rejects_destination_aliases_before_writing(tmp_path, names):
    archive=tmp_path/'images.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for name in names:
            z.writestr(name,'fixture')
    with pytest.raises(ValueError):
        extract_images(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_zip_extract_rejects_traversal_and_overwrite(tmp_path):
    archive = tmp_path/'images.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../outside.txt','no')
    with pytest.raises(ValueError,match='Unsafe'):
        extract_images(archive,tmp_path/'out')
    assert not (tmp_path/'out').exists() and not (tmp_path/'outside.txt').exists()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('images/test.txt','fixture')
    extract_images(archive,tmp_path/'out')
    assert (tmp_path/'out/images/test.txt').read_text()=='fixture'
    with pytest.raises(FileExistsError):
        extract_images(archive,tmp_path/'out')


def test_download_pins_revision_and_explicit_file_scope(tmp_path, monkeypatch):
    import huggingface_hub
    called = []
    monkeypatch.setattr(huggingface_hub.HfApi,'list_repo_files',lambda *a,**k:['test/data/a.jsonl','train/image/a.png'])
    def download(*args,**kwargs):
        called.append(kwargs)
        output=Path(kwargs['local_dir'])
        for name in kwargs['allow_patterns']:
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('fixture')
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    with pytest.raises(ValueError,match='40-hex'):
        download_dataset('gui360','main',tmp_path/'out',include=['test/*'])
    with pytest.raises(ValueError,match='--include'):
        download_dataset('gui360','a'*40,tmp_path/'out')
    with pytest.raises(ValueError,match='matches no'):
        download_dataset('gui360','a'*40,tmp_path/'out',include=['bad/*'])
    report=download_dataset('gui360','a'*40,tmp_path/'out',include=['test/*'])
    assert report['files']==['test/data/a.jsonl']
    assert called[0]['revision']=='a'*40 and called[0]['repo_type']=='dataset'
