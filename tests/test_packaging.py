"""Default configuration survives source and wheel distribution builds."""
from importlib.resources import files
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

from gui_joint_control.cli import load_config


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = 'gui_joint_control/configs/naacl_reference.json'


def test_packaged_default_preserves_the_explicit_reference_config():
    # The explicit path remains available to existing commands and registries.
    reference = ROOT / 'configs/naacl_reference.json'
    assert files('gui_joint_control').joinpath('configs/naacl_reference.json').read_bytes() == reference.read_bytes()
    assert load_config() == load_config(reference)


def test_wheel_and_sdist_ship_default_config_and_wheel_smoke_runs(tmp_path):
    if importlib.util.find_spec('setuptools') is None:
        pytest.skip('Distribution builds require the declared setuptools build backend')
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in'):
        shutil.copy2(ROOT / name, source / name)
    for name in ('src', 'configs'):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    env = {**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'PYTHONDONTWRITEBYTECODE':'1'}
    build = subprocess.run([sys.executable, '-c',
        'import sys; destination = sys.argv[1]; from setuptools.build_meta import build_wheel, build_sdist; '
        'build_wheel(destination); build_sdist(destination)', str(artifacts)],
        cwd=source, env=env, capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    expected = (ROOT / 'configs/naacl_reference.json').read_bytes()
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(next(artifacts.glob('*.whl'))) as wheel:
        assert wheel.read(RESOURCE) == expected
        wheel.extractall(installed)
    with tarfile.open(next(artifacts.glob('*.tar.gz'))) as sdist:
        prefix = sdist.getnames()[0].split('/')[0]
        assert sdist.extractfile(prefix + '/src/' + RESOURCE).read() == expected
        assert sdist.extractfile(prefix + '/configs/naacl_reference.json').read() == expected
    output = tmp_path / 'smoke'
    smoke = subprocess.run([sys.executable, '-I', '-c',
        'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); '
        'from gui_joint_control import cli; '
        'assert Path(cli.__file__).is_relative_to(Path(sys.argv[1])); '
        'cli.main(["smoke", "--output", sys.argv[2], "--updates", "1"])',
        str(installed), str(output)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
    assert json.loads((output / 'software_fixture_config.json').read_text(encoding='utf-8')) == json.loads(expected)
    report = json.loads((output / 'SMOKE_ONLY.json').read_text(encoding='utf-8'))
    assert report['checkpoint_exists'] and report['finite_training_steps'] == 1
