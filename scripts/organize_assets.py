#!/usr/bin/env python3
"""Safely put shared pretrained weights and optional dataset backup in artifacts/.

Default is read-only. --apply copies weights (checksum-verified) and replaces
collector/models binaries with relative links to the verified central files.
Does not delete legacy data, recordings, or source models.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

MODEL_NAMES = ('yolo11m-pose.pt', 'yolo11n-pose.pt', 'pose_landmarker_full.task')


def sha256(file: Path) -> str:
    hasher = hashlib.sha256()
    with file.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def checked_copy(source: Path, destination: Path) -> None:
    """Copy without overwriting differing assets; no leftover temporary files."""
    if destination.is_symlink():
        raise RuntimeError(f'Refusing symlink destination: {destination}')
    if destination.is_file():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f'Existing asset differs, refusing overwrite: {destination}')
        print(f'[OK] Already backed up: {destination.name}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.copy-', delete=False) as output:
            temp_path = Path(output.name)
            size = source.stat().st_size
            copied = 0
            progress_step = -1
            with source.open('rb') as reader:
                while chunk := reader.read(4 * 1024 * 1024):
                    output.write(chunk)
                    copied += len(chunk)
                    step = int(20 * copied / size) if size else 20
                    if size >= 100 * 1024 * 1024 and sys.stderr.isatty() and step != progress_step:
                        progress_step = step
                        filled = min(step, 20)
                        print(f'\rCopying {source.name}: [{"#" * filled}{"-" * (20-filled)}] '
                              f'{copied / (1024**2):.0f}/{size / (1024**2):.0f} MiB',
                              file=sys.stderr, end='', flush=True)
            if progress_step >= 0:
                print(file=sys.stderr)
            output.flush()
            os.fsync(output.fileno())
        if sha256(source) != sha256(temp_path):
            raise RuntimeError(f'Checksum failed during copy: {source.name}')
        # Never intentionally overwrite an existing different backup.
        if destination.exists():
            if sha256(source) != sha256(destination):
                raise RuntimeError(f'Existing asset differs, refusing overwrite: {destination}')
        else:
            temp_path.replace(destination)
        print(f'[OK] Copied and verified: {destination.name}')
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def plan(root: Path, backup_dataset: bool):
    collector = root / 'collector' / 'models'
    central = root / 'artifacts' / 'pretrained'
    legacy = root / 'PoseForge_Real_Posture_Collector_V1'
    models = []
    for name in MODEL_NAMES:
        live = collector / name
        shared = central / name
        old = legacy / 'models' / name
        expected_link = Path('../../artifacts/pretrained') / name
        if live.is_symlink() and os.readlink(live) != str(expected_link):
            raise RuntimeError(f'Unexpected collector symlink: {live}')
        files = [path for path in (shared, live, old) if path.is_file()]
        if not files:
            raise RuntimeError(f'Missing weight {name} in collector/, artifacts/, and legacy/')
        checksums = {sha256(path) for path in files}
        if len(checksums) != 1:
            raise RuntimeError(f'Different weights with the same name: {name}; aborting')
        if shared.is_symlink():
            raise RuntimeError(f'Refusing central symlink: {shared}')
        source = shared if shared.is_file() else (live if live.is_file() else old)
        models.append((name, source, live, shared, expected_link))
    dataset = None
    if backup_dataset:
        archive = legacy / 'Real_Posture_Dataset.zip'
        target = root / 'artifacts' / 'dataset_backup' / archive.name
        if not archive.is_file():
            if not target.is_file():
                raise RuntimeError('Dataset ZIP not found in legacy/ or artifacts/dataset_backup/')
            print('[OK] Dataset backup already exists; no legacy ZIP required.')
        else:
            if not zipfile.is_zipfile(archive):
                raise RuntimeError(f'Dataset archive does not look like a ZIP: {archive}')
            if target.exists() and sha256(archive) != sha256(target):
                raise RuntimeError('Dataset backup differs from legacy ZIP; refusing overwrite')
            dataset = (archive, target)
    return models, dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Copy checked assets and link collector weights')
    mode.add_argument('--verify', action='store_true', help='Require completed shared layout')
    parser.add_argument('--backup-dataset', action='store_true', help='Preserve existing dataset ZIP in ignored artifacts/')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    models, dataset = plan(root, args.backup_dataset)
    for name, source, live, shared, link in models:
        if args.apply:
            if not shared.is_file():
                checked_copy(source, shared)
            if not live.is_symlink():
                live.parent.mkdir(parents=True, exist_ok=True)
                temp_link = live.parent / f'.{name}.link-{os.getpid()}'
                try:
                    temp_link.symlink_to(link)
                    if sha256(temp_link) != sha256(shared):
                        raise RuntimeError(f'Link checksum mismatch: {name}')
                    temp_link.replace(live)
                finally:
                    temp_link.unlink(missing_ok=True)
            print(f'[OK] collector/models/{name} → artifacts/pretrained/{name}')
        elif args.verify:
            if not shared.is_file() or not live.is_symlink() or live.resolve() != shared.resolve():
                raise RuntimeError(f'Shared model layout incomplete: {name}; use --apply')
            print(f'[OK] Shared model verified: {name}')
        else:
            print(f'[PLAN] {name}: {source.relative_to(root)} → artifacts/pretrained/, collector link')
    if dataset:
        archive, target = dataset
        if args.apply:
            checked_copy(archive, target)
        elif args.verify:
            if not target.is_file() or sha256(archive) != sha256(target):
                raise RuntimeError('Dataset backup missing or differs; use --apply --backup-dataset')
            print('[OK] Local dataset ZIP checksum verified')
        else:
            print(f'[PLAN] {archive.relative_to(root)} → {target.relative_to(root)}')
    if not args.apply and not args.verify:
        print('[PLAN] No files changed. Use --apply --backup-dataset to perform migration.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f'[FAIL] {exc}') from exc
