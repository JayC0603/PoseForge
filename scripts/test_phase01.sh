#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
export PYTHONDONTWRITEBYTECODE=1
# ROS Humble injects Python 3.10 packages and pytest plugins into the shell.
# Keep Conda Python 3.11 tests independent of the ROS Python environment.
unset PYTHONPATH PYTHONHOME PYTEST_ADDOPTS PYTEST_PLUGINS
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$ROOT/collector"
# Syntax-check without compileall: compileall generates __pycache__ inside the project.
"$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -B -c '
from pathlib import Path
paths = [Path("main.py"), Path("process_dataset.py"), Path("audit_recordings.py")]
paths += sorted(Path("posture_collector").rglob("*.py"))
for path in paths:
    compile(path.read_bytes(), str(path), "exec")
print(f"[OK] Syntax: {len(paths)} Python files")'
# Disable pytest cacheprovider. -B and PYTHONDONTWRITEBYTECODE prevent bytecode cache.
"$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" python -B -m pytest -p no:cacheprovider -q \
  tests/test_append_existing_dataset.py tests/test_storage_and_capture.py tests/test_pipeline_cli.py tests/test_phase01_contract.py tests/test_asset_layout.py tests/test_runtime_layout.py
echo '[OK] Required collector regression checks complete.'
