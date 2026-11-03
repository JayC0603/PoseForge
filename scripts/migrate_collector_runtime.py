#!/usr/bin/env python3
"""Move the existing local collector dataset/config to the ignored artifacts area.

Safe by default: use --apply to move, never overwrite an existing destination.
No copying, deleting data or generated temporary files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def migrate(project: Path, apply: bool) -> int:
    old = project / 'collector' / 'Real_Posture_Dataset'
    new = project / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
    if old.is_symlink() or new.is_symlink():
        raise RuntimeError('Symlinked dataset root: refusing to move')
    if not old.exists():
        if new.is_dir():
            print('[OK] Already using:', new)
        else:
            print('[OK] No existing collector dataset to migrate; first launch will initialize:', new)
        return 0
    if not old.is_dir():
        raise RuntimeError(f'Expected directory: {old}')
    if new.exists():
        raise RuntimeError(f'Both dataset locations exist. Refusing merge/overwrite: {new}')
    files = sum(1 for path in old.rglob('*') if path.is_file())
    print(f'[PLAN] Move {files} files (config + recordings)')
    print('  From:', old)
    print('    To:', new)
    if not apply:
        print('[PLAN] No changes. Use --apply after closing collector.')
        return 0
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)  # same project filesystem; no duplicate data
    print(f'[OK] Moved {files} files; all paths under the dataset root are unchanged')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        return migrate(args.root.resolve(), args.apply)
    except (RuntimeError, OSError) as exc:
        print('[BLOCKED]', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
