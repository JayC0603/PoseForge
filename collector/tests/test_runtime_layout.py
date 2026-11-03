"""Runtime path migration and camera persistence without video or private data."""
import json
import subprocess
import sys
from pathlib import Path

from posture_collector.storage import DataStore, DEFAULT_COLLECTOR_DATA_ROOT

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / 'scripts' / 'migrate_collector_runtime.py'


def run(root, *args):
    return subprocess.run([sys.executable, '-B', str(TOOL), '--root', str(root), *args],
                          capture_output=True, text=True)


def test_camera_url_persists_across_app_restarts(tmp_path):
    root = tmp_path / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
    store = DataStore(root)
    cams = store.cameras()
    cams[0]['url'] = 'http://192.0.2.10:4747/video'
    cams[1]['url'] = 'http://192.0.2.11:4747/video'
    store.set_cameras(cams)
    reopened = DataStore(root)
    assert reopened.cameras() == cams
    assert json.loads((root / 'config' / 'cameras.json').read_text()) == cams


def test_migration_moves_config_and_recordings_without_changes(tmp_path):
    old = tmp_path / 'collector' / 'Real_Posture_Dataset'
    (old / 'config').mkdir(parents=True)
    (old / 'recordings' / 'TEST').mkdir(parents=True)
    (old / 'config' / 'cameras.json').write_text('{"ip":"example"}')
    (old / 'recordings' / 'TEST' / 'sample.bin').write_bytes(b'video bytes')
    assert run(tmp_path).returncode == 0
    assert old.exists() and not (tmp_path / 'artifacts').exists()
    result = run(tmp_path, '--apply')
    assert result.returncode == 0, result.stderr
    new = tmp_path / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
    assert not old.exists()
    assert (new / 'config' / 'cameras.json').read_text() == '{"ip":"example"}'
    assert (new / 'recordings' / 'TEST' / 'sample.bin').read_bytes() == b'video bytes'
    assert run(tmp_path, '--apply').returncode == 0


def test_migration_blocks_existing_destination(tmp_path):
    old = tmp_path / 'collector' / 'Real_Posture_Dataset'
    new = tmp_path / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / 'important').write_bytes(b'do not delete')
    result = run(tmp_path, '--apply')
    assert result.returncode != 0 and '[BLOCKED]' in result.stderr
    assert (old / 'important').read_bytes() == b'do not delete'


def test_default_runtime_root_points_outside_collector():
    assert DEFAULT_COLLECTOR_DATA_ROOT == REPO / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
