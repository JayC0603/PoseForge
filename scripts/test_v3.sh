#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
# Prevent ROS Humble Python 3.10 pytest plugins leaking into Conda Python 3.11.
unset PYTHONPATH PYTHONHOME PYTEST_ADDOPTS PYTEST_PLUGINS
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/collector"
tests=(
  tests/test_v3_reextract_and_split.py
  tests/test_replay_and_full_reextract.py
  tests/test_feature_v2.py
  tests/test_dataset_audit.py
  tests/test_storage_and_capture.py
  tests/test_pipeline_cli.py
  tests/test_phase01_contract.py
  tests/test_asset_layout.py
  tests/test_runtime_layout.py
)
if [[ -f tests/test_append_existing_dataset.py ]]; then
  tests+=(tests/test_append_existing_dataset.py)
else
  echo '[WARN] Append-safe regression test absent; check collector/storage.py before recording.'
fi
"$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" \
  python -B -m pytest -p no:cacheprovider -q "${tests[@]}"
if [[ "${1:-}" == '--ui' ]]; then
  "$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" \
    python -B -m pytest -p no:cacheprovider -q tests/test_ui.py
fi
