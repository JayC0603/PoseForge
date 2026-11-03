# PoseForge Collector (Ubuntu 22.04)

Dataset collector V1 with PySide6 UI, IP/USB cameras, YOLO-Pose/MediaPipe,
ByteTrack, raw MP4 and original keypoints format. Collector UI and recording
schema are kept compatible with the original Windows implementation.

From the project root:

```bash
conda activate lab2_imp
bash scripts/run_collector.sh
bash scripts/test_phase01.sh
```

Data and camera configuration live at
`artifacts/collector_runtime/Real_Posture_Dataset/`. Existing configurations
under `collector/Real_Posture_Dataset/` must be moved once using
`python -B scripts/migrate_collector_runtime.py --apply` (collector closed).

Shared pretrained weights: `artifacts/pretrained/`. The paths under
`collector/models/` are symbolic links for legacy compatibility.

`process_dataset.py`, `audit_recordings.py` and `tools/` are retained
for verification and reproducibility. Train/val/test dataset creation for
Lab 2 occurs on Colab, **not** on the laptop's original dataset.

More instructions are in the local-only `docs/RUN.md` (not Git tracked).
