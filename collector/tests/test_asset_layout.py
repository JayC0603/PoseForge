"""Shared weight migration, using small synthetic files (no private data)."""
import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
TOOL = PROJECT / 'scripts' / 'organize_assets.py'
NAMES = ('yolo11m-pose.pt', 'yolo11n-pose.pt', 'pose_landmarker_full.task')


def run(root, *args):
    return subprocess.run([sys.executable, '-B', str(TOOL), '--root', str(root), *args],
                          text=True, capture_output=True)


def fake_layout(root):
    source = root / 'collector' / 'models'
    source.mkdir(parents=True)
    for name in NAMES:
        (source / name).write_bytes((name * 5).encode())
    old = root / 'PoseForge_Real_Posture_Collector_V1'
    old.mkdir()
    with zipfile.ZipFile(old / 'Real_Posture_Dataset.zip', 'w') as z:
        z.writestr('labels.csv', 'clip_id,person_id,camera_view,label\n')
    return source, old


def test_shared_assets_copy_link_and_verify(tmp_path):
    live, legacy = fake_layout(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in live.iterdir()}
    zip_digest = hashlib.sha256((legacy / 'Real_Posture_Dataset.zip').read_bytes()).hexdigest()
    plan = run(tmp_path, '--backup-dataset')
    assert plan.returncode == 0, plan.stderr
    assert not (tmp_path / 'artifacts').exists()
    migrated = run(tmp_path, '--apply', '--backup-dataset')
    assert migrated.returncode == 0, migrated.stderr
    assert run(tmp_path, '--verify', '--backup-dataset').returncode == 0
    assert run(tmp_path, '--apply', '--backup-dataset').returncode == 0  # idempotent
    for name in NAMES:
        linked = live / name
        central = tmp_path / 'artifacts' / 'pretrained' / name
        assert linked.is_symlink()
        assert linked.resolve() == central.resolve()
        assert hashlib.sha256(central.read_bytes()).hexdigest() == before[name]
    assert hashlib.sha256((tmp_path / 'artifacts' / 'dataset_backup' / 'Real_Posture_Dataset.zip').read_bytes()).hexdigest() == zip_digest
    assert (legacy / 'Real_Posture_Dataset.zip').is_file()  # legacy untouched
    assert not list(tmp_path.rglob('.copy-*'))
    assert not list(tmp_path.rglob('.*.link-*'))


def test_mismatch_rejected_before_mutating(tmp_path):
    live, _ = fake_layout(tmp_path)
    central = tmp_path / 'artifacts' / 'pretrained'
    central.mkdir(parents=True)
    (central / NAMES[0]).write_bytes(b'different')
    result = run(tmp_path, '--apply', '--backup-dataset')
    assert result.returncode != 0
    assert (live / NAMES[0]).is_file() and not (live / NAMES[0]).is_symlink()
    assert not (central / NAMES[1]).exists()
    assert not (tmp_path / 'artifacts' / 'dataset_backup').exists()


def test_verify_rejects_unmigrated_files(tmp_path):
    fake_layout(tmp_path)
    result = run(tmp_path, '--verify')
    assert result.returncode != 0
