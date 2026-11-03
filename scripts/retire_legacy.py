#!/usr/bin/env python3
"""Verify legacy collector/data against active copies before retiring it.

No output is written to the project during verification. Run with --delete only
when the ZIP backup contains every byte of the legacy dataset and collector
source/model files exactly match their current copies.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, BadZipFile

LEGACY_NAME = "PoseForge_Real_Posture_Collector_V1"
MODEL_NAMES = {"yolo11m-pose.pt", "yolo11n-pose.pt", "pose_landmarker_full.task"}
OLD_DOCS = {"TRACKING.md", "UI_REDESIGN.md", "QUICK_START_VI.txt"}
GENERATED_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
GENERATED_FILES = {".DS_Store"}


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def same_file(left: Path, right: Path) -> bool:
    if not right.is_file() or left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        return sha256_stream(a) == sha256_stream(b)


def ignored_generated(relative: Path) -> bool:
    return (any(part in GENERATED_DIRS for part in relative.parts)
            or relative.name in GENERATED_FILES
            or relative.suffix in {".pyc", ".pyo"})


def verify_weights_and_source(root: Path, legacy: Path) -> tuple[int, int]:
    collector = root / "collector"
    models = root / "artifacts" / "pretrained"
    if not collector.is_dir() or not models.is_dir():
        raise RuntimeError("Missing active collector or artifacts/pretrained")
    for name in sorted(MODEL_NAMES):
        active_model = models / name
        collector_model = collector / "models" / name
        if not active_model.is_file() or not collector_model.is_file():
            raise RuntimeError(f"Missing active model: {name}")
        if not collector_model.is_symlink() or collector_model.resolve() != active_model.resolve():
            raise RuntimeError(f"Model must be shared through collector symlink: {name}")

    checked_source = 0
    checked_models = 0
    for path in legacy.rglob("*"):
        relative = path.relative_to(legacy)
        if ignored_generated(relative):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Unexpected symlink in legacy tree: {relative}")
        if path.is_dir():
            continue
        if relative.parts[0] == "Real_Posture_Dataset":
            continue
        if relative.name == "Real_Posture_Dataset.zip" and len(relative.parts) == 1:
            backup = root / "artifacts" / "dataset_backup" / relative.name
            if not same_file(path, backup):
                raise RuntimeError("Remaining legacy dataset ZIP differs from artifacts backup")
            continue
        if relative.parts[0] == "models":
            if len(relative.parts) != 2 or relative.name not in MODEL_NAMES:
                raise RuntimeError(f"Unexpected model asset: {relative}")
            target = models / relative.name
            checked_models += 1
        else:
            target = collector / (Path("docs") / relative if relative.name in OLD_DOCS and len(relative.parts) == 1 else relative)
            checked_source += 1
        if not same_file(path, target):
            raise RuntimeError(f"Missing/different source file: {relative}")
    if checked_source == 0:
        raise RuntimeError("Legacy source unexpectedly empty; refusing deletion")
    print(f"[OK] Legacy source: {checked_source} identical files; model weights: {checked_models}")
    return checked_source, checked_models


def archive_relative(name: str) -> Path:
    name = name.replace("\\", "/").strip("/")
    parts = name.split("/")
    if "Real_Posture_Dataset" in parts:
        parts = parts[parts.index("Real_Posture_Dataset") + 1:]
    if not parts or not all(part not in ("", ".", "..") for part in parts):
        raise RuntimeError(f"Invalid ZIP entry: {name!r}")
    if parts[0] not in {"config", "recordings", "processed", "reports", "splits", "labels.csv"}:
        raise RuntimeError(f"Unexpected ZIP layout/entry: {name!r}")
    return Path(*parts)


def verify_dataset(root: Path, legacy: Path) -> int:
    disk = legacy / "Real_Posture_Dataset"
    backup = root / "artifacts" / "dataset_backup" / "Real_Posture_Dataset.zip"
    if not disk.is_dir() or disk.is_symlink():
        raise RuntimeError("Legacy raw dataset missing; use backup move instead of deletion")
    if not backup.is_file() or backup.is_symlink():
        raise RuntimeError("Dataset ZIP backup missing/not a regular file")
    disk_files: dict[Path, Path] = {}
    for path in disk.rglob("*"):
        relative = path.relative_to(disk)
        if path.is_symlink():
            raise RuntimeError(f"Dataset contains symlink: {relative}")
        if path.is_file() and not ignored_generated(relative):
            disk_files[relative] = path
    with ZipFile(backup, "r") as archive:
        zip_files = {}
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                raise RuntimeError(f"ZIP contains symlink: {entry.filename}")
            relative = archive_relative(entry.filename)
            if ignored_generated(relative):
                continue
            if relative in zip_files:
                raise RuntimeError(f"Duplicate ZIP entry: {relative}")
            zip_files[relative] = entry
        if disk_files.keys() != zip_files.keys():
            missing = sorted(str(x) for x in (disk_files.keys() - zip_files.keys()))
            extra = sorted(str(x) for x in (zip_files.keys() - disk_files.keys()))
            raise RuntimeError(f"Dataset ZIP does not match raw folder: files only on disk={missing[:5]}, only in ZIP={extra[:5]}")
        if not disk_files:
            raise RuntimeError("Dataset contains no files")
        for index, (relative, local_path) in enumerate(sorted(disk_files.items()), 1):
            entry = zip_files[relative]
            if local_path.stat().st_size != entry.file_size:
                raise RuntimeError(f"Size mismatch: {relative}")
            with local_path.open("rb") as a, archive.open(entry) as b:
                if sha256_stream(a) != sha256_stream(b):
                    raise RuntimeError(f"Data mismatch: {relative}")
            if index % 200 == 0 or index == len(disk_files):
                print(f"\r[CHECK] ZIP versus raw dataset: {index}/{len(disk_files)} files", end="", flush=True)
        print()
    print(f"[OK] Dataset backup matches all {len(disk_files)} raw files")
    return len(disk_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--delete", action="store_true", help="Verify everything, then delete legacy after typed confirmation")
    group.add_argument("--move-to-backup", action="store_true", help="Move legacy unchanged to ignored artifacts/legacy_backup/")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    legacy = root / LEGACY_NAME
    if legacy.is_symlink() or not legacy.is_dir():
        raise RuntimeError(f"Legacy folder not found or unsafe: {legacy}")
    if args.move_to_backup:
        destination = root / "artifacts" / "legacy_backup" / LEGACY_NAME
        if destination.exists():
            raise RuntimeError(f"Backup destination already exists: {destination}")
        if not (root / "artifacts" / "dataset_backup" / "Real_Posture_Dataset.zip").is_file():
            raise RuntimeError("Dataset ZIP backup missing; refusing to move")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(legacy).st_dev != os.stat(destination.parent).st_dev:
            raise RuntimeError("Cannot atomically move across devices")
        legacy.rename(destination)
        print(f"[OK] Legacy moved without copying: {destination}")
        return 0
    verify_weights_and_source(root, legacy)
    verify_dataset(root, legacy)
    if not args.delete:
        print("[READY] All checks passed; use --delete to remove legacy, or --move-to-backup to keep it")
        return 0
    response = input(f"Type DELETE to permanently remove {LEGACY_NAME}: ").strip()
    if response != "DELETE":
        print("[STOP] No files deleted")
        return 1
    shutil.rmtree(legacy)
    print(f"[OK] Removed legacy directory: {legacy}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, BadZipFile) as error:
        print(f"[BLOCKED] {error}", file=sys.stderr)
        sys.exit(2)
