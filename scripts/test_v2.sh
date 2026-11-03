#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/conda_common.sh"
unset PYTHONPATH PYTHONHOME PYTEST_ADDOPTS PYTEST_PLUGINS
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/collector"
files=(tests/test_feature_v2.py tests/test_dataset_audit.py
       tests/test_phase01_contract.py tests/test_storage_and_capture.py
       tests/test_pipeline_cli.py tests/test_asset_layout.py tests/test_runtime_layout.py)
if [[ -f tests/test_append_existing_dataset.py ]]; then
  files+=(tests/test_append_existing_dataset.py)
fi
"$CONDA_CMD" run --no-capture-output -n "$POSEFORGE_CONDA_ENV" \
    python -B -m pytest -p no:cacheprovider -q "${files[@]}"
