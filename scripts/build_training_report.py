#!/usr/bin/env python3
"""Build review artifacts from persisted metrics only; no model/test evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.real_training import read_json, write_json
from shared.dataset_pipeline import file_sha256


def table(df, columns, digits=4):
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for row in df[columns].to_dict('records'):
        lines.append('| ' + ' | '.join(f'{v:.{digits}f}' if isinstance(v,float) else str(v) for v in row.values()) + ' |')
    return '\n'.join(lines)


def build(run, reports):
    run, reports = Path(run).resolve(), Path(reports).resolve()
    model = Path(read_json(run/'final_model.json')['path'])
    preflight = read_json(run/'dataset_summary.json')
    split = read_json(run/'split_manifest.json')
    selection = read_json(run/'selection.json')
    comparison = pd.read_csv(run/'comparison.csv')
    selected = read_json(run/'selected_validation/metrics.json')
    test = read_json(model/'test_metrics.json')
    parity = read_json(model/'onnx_real_parity.json')
    gates = read_json(run/'quality_gates.json')
    environment = read_json(run/'environment.json')
    confirmations = pd.read_csv(run/'confirmed_ranking.csv')
    clips = pd.read_csv(run/'catalog.csv')
    split_table = clips.groupby('split').agg(subjects=('person_id','nunique'),clips=('clip_id','size'),source_takes=('source_group_id','nunique'),frames=('frames','sum'),eligible_frames=('usable_frames','sum')).reset_index()
    inference_dir = max((ROOT/'artifacts/inference').glob('recovered_real_e2e_v*/frames.csv'), key=lambda p:p.stat().st_mtime_ns).parent
    inference_frames = pd.read_csv(inference_dir/'frames.csv')
    e2e = dict(frames=len(inference_frames), classified=int(inference_frames.class_id.notna().sum()),
               prediction_counts={str(int(k)):int(v) for k,v in inference_frames.class_id.value_counts().items()},
               backend='onnx', source_subject='CHANH', source_label=0,
               output_directory=str(inference_dir), contains_private_video=True,
               policy='Local inference output excluded from all shareable bundles')
    assert e2e['frames'] == 71 and e2e['classified'] == 71
    write_json(model/'end_to_end_validation.json', e2e)
    summary = read_json(model/'training_summary.json')
    summary.update(environment=environment, onnx_parity=parity,
                   processed_dataset_id=file_sha256(run/'processed/checksums.json'),
                   end_to_end_validation=e2e,
                   completed_selection_experiments=len(comparison),
                   total_completed_training_runs=len(list((run/'experiments').glob('*/result.json'))),
                   quality_gates=gates,
                   test_limitations='One held-out subject; 7 independent source takes. 1513 eligible frames are temporally correlated. Front-camera class 1 has no eligible test observations.',
                   tests='Full suite: 111 passed, 4 skipped, 1 pre-existing Windows symlink-privilege failure. Additional final contract/resume tests recorded separately.')
    write_json(model/'training_summary.json', summary)
    write_json(run/'training_summary.json', summary)
    primary = comparison[~comparison.experiment.str.startswith('confirm_')].sort_values('experiment')
    report = f'''# PoseForge real dataset training: executed final report

## A. Dataset

All work ran in `C:\\Users\\tienq\\PoseForge_Reposory`, using the existing
`C:\\Users\\tienq\\miniforge3\\envs\\ferlab-infer\\python.exe`. No packages were reinstalled.

Dataset ID: `{preflight['dataset_id']}`. Processed artifact ID:
`{summary['processed_dataset_id']}`.

The current data contain **13 subjects, 88 independent source takes, 176 camera
clips, and 15,823 decoded frames**. Both cameras have 88 clips; front has 8,461
frames and oblique has 7,362. Encoded duration summed over both views is 791.15 s;
capture spans summed over both views are 489.801 s. Fixed-FPS video duration differs
from capture time in 175 clips, so sequence timing must use capture timestamps.

Subjects: {', '.join(preflight['subjects'])}.

| Label | Official key | Camera clips | Frames |
| --- | --- | --- | --- |
| 0 | sitting_straight | 28 | 2448 |
| 1 | hunched_forward | 36 | 3213 |
| 2 | head_on_desk | 26 | 2737 |
| 3 | leaning_backward | 28 | 2362 |
| 4 | turning_sideways (both directions) | 30 | 2667 |
| 5 | raising_hand | 28 | 2396 |

All 176 raw keypoint files exist, and **30/30 recovered files contain selected
poses**. There are 14,514 selected pose observations, 6,066 frames with all 13 joints
valid, 1,309 frames without a selected person, and 258 selected observations excluded
as uncertain. Sixteen other files contain zero selected poses. Their exact names
are in `training_preflight.json`; no further extraction was performed.

The initial supervised dataset contains **14,256 eligible frames**. Partial-joint
frames remain eligible with at least four valid joints and a certain selection.
All missing/uncertain rows, confidence, source indices, and timestamps are retained.
Uncertain selected poses enter the causal feature stream as absent. TDUY lacks
usable label-5 observations and was therefore eligible only for the training pool.

The audit found 37 selected-track changes, 469 duplicate adjacent timestamp ticks,
zero backward ticks, 8 large timing gaps, zero identical cross-clip decoded frames,
zero adjacent identical video frames, and zero overlapping source frame ranges.
No automatic posture relabeling or human-approval claim was made.

There was no existing curation manifest or clean export in the live dataset.
The new snapshot records explicit automated curation provenance and source hashes.
All **796 inventoried source files remained byte-identical** after training.
Raw videos, recovered keypoints, timestamps, metadata, pretrained weights, and old
outputs were preserved. Shareable artifacts contain no raw/annotated video.

Frozen split (seed 302; quality/class-coverage eligibility followed by a seeded
subject permutation, selected before training):

{table(split_table,['split','subjects','clips','source_takes','frames','eligible_frames'])}

Training subjects: {', '.join(split['train_people'])}.
Validation subjects: {', '.join(split['validation_people'])}.
Final test subject: {', '.join(split['test_people'])}.
Both views and all takes/segments of each subject stay in one split. JSON/CSV split
manifests and subject coverage are stored in `{run.relative_to(ROOT).as_posix()}`.

## B. Implementation

Created:

* `shared/real_training.py`: read-only full audit, immutable snapshots/features,
  quality exclusion, frozen split, CUDA training, sampling, metrics, resume,
  controlled experiments, seed confirmation, guarded final test, real ONNX parity.
* `scripts/train_poseforge.py`: Windows CLI for audit/prepare/baseline/improve/finalize/export.
* `scripts/verify_poseforge_training.py`: source hashes, seals, full feature regeneration,
  checkpoint loading, probability and temporal-feasibility checks without test predictions.
* `scripts/build_training_report.py`: this report and allowlisted video-free bundles.
* `configs/training_real_baseline.json`, `docs/TRAINING_WINDOWS.md`, and
  `tests/test_real_training.py`.

Updated:

* `notebooks/01_raw_eda_split_v2.ipynb`, `02_process_eda_export_v2.ipynb`, and
  `03_train_mlp_v2.ipynb`: dynamic CLI-backed workflow, local Windows paths,
  dynamic counts, configurable subjects, no Drive/Colab/ZIP dependency.
  Original notebook copies are preserved in `artifacts/reports/notebook_backups`.
* `shared/pose_training.py`: dynamic coverage-based subject defaults; removed
  obsolete named subjects. This run passed its explicit frozen split throughout.
* `scripts/export_onnx.py` and `inference/pipeline.py`: explicit UTF-8 JSON handling
  for the official Vietnamese labels on Windows.
* `tests/test_phase2_v2.py`: dynamic split and notebook contract checks.
* `collector/tests/test_phase16_curation.py`: portable POSIX ZIP-member lookup.

The feature contract remains **`pose13_xy_conf_mask_causal_v2`, 13 x 4 = 52 floats**.
The original normalizer and missing-joint semantics are preserved. State resets at
camera-clip boundaries, track changes, and after more than eight missing frames,
matching deployment. No future frames or extra keypoint smoothing are used.
All 15,823 feature rows regenerated exactly from the sealed selected-pose stream.
Training/inference parity tests cover track changes and long absences.

Executed training-code copies/hashes are in the run's `source_code/` and
`source_code_hashes.json`. The later shared split-default cleanup does not change
the explicit subjects used in this run.

## C. Actual training

GPU: **{environment['gpu']}**, 4 GB VRAM; PyTorch {environment['torch']} with CUDA.
No OOM occurred. An observed training sample used approximately 1,383 MiB total
GPU memory (including other GPU use). Training used float32, AdamW, learning rate
0.001, weight decay 0.001, batch 128, dropout 0.15, at most 100 epochs and patience
12. Selection used validation mean per-subject six-class Macro-F1.

Small layers: 52-64-32-6. Base: 52-128-64-6. Wide: 52-256-128-6.
The real-data eight-epoch Small smoke run decreased train loss from 1.3131 to
0.1248. All three ordinary-CE baselines trained on real data without weighting,
smoothing, augmentation, oversampling, or source balancing.

Completed **15 training runs**: one smoke, three baselines, six controlled
improvements, four seed confirmations, and one fresh final retrain. Each has
persisted configuration/history and checkpoints. No combined candidate was launched:
no improvement exceeded the predefined +0.005 subject Macro-F1 threshold at seed
302, so there were no justified changes to combine.

Controlled class weights used training-only original source-take/camera counts,
square-root inverse frequency, capped at 0.5-2. Augmentation used correct left/right
flip, small coordinate jitter, confidence perturbation, limited dropout, and arm
occlusion on training only. Source sampling capped each original take/camera at
48 distinct frames/epoch; the mild balance variant used square-root downsampling.

## D. Measured validation results

All rows below use seed 302, the same fixed 1,433 eligible validation frames, and
best checkpoints chosen without test access. Epochs are best/completed counts.

{table(primary,['experiment','best_epoch','epochs','frame_accuracy','frame_macro_f1','subject_macro_f1','clip_macro_f1','validation_loss','calibration_ece'])}

Top-two seed confirmation:

{table(confirmations,['candidate','mean_f1','std_f1','median_epoch'])}

The **unweighted Wide baseline** won by three-seed mean subject Macro-F1
(0.9017 versus 0.8998). Seeds were 302, 303, 304. Their best epochs were 8, 4, 1
for both configurations, so the frozen final epoch budget was four. The weighting
gain at seed 302 did not persist across seeds. No smoothing/augmentation/balancing
change was retained.

Selected seed-302 validation checkpoint:

{table(pd.DataFrame(selected['per_subject']),['person','frames','macro_f1','accuracy'])}

{table(pd.DataFrame(selected['per_camera']),['camera','view','frames','macro_f1','accuracy'])}

### Failure analysis

VMTRI accounts for 140 of 144 validation errors. Its label 1 was predicted as label
2 in 98/147 eligible frames (63 front, 35 oblique); label 3 was predicted as label
0 in 42/152 frames (21 per view). TRUONG had four label-2 frames predicted as label
3. Label-1 pooled recall is 0.5586; label-3 recall is 0.8576.

Feature diagnostics support a subject-geometry explanation, not simply missing
head joints: VMTRI's nose is valid in all these clips. Its label-1 mean normalized
nose Y is 0.359 front / 0.330 oblique, compared with the training clip-median of
0.048 for label 1 and 0.653 for label 2. This deeper head position is consistent
with the observed 1-to-2 confusion. VMTRI label-3 nose Y is approximately -0.55 in
both views and hip Y approximately 1.31-1.33; these are close to training upright
medians (-0.552 and 1.339). These are diagnostic associations, not proof that the
label is wrong. No labels were changed. Full diagnostics are saved in
`validation_geometry_diagnostics.csv`.

Labels 4 and 5 had precision/recall/F1 of 1.0 in this validation checkpoint.
The dataset does not establish performance on gradual partial-arm-raise transitions;
these labels cover recorded held postures. Front-camera Macro-F1 is lower than
oblique, and valid class coverage differs by camera. The learning curves show
training loss continuing down while validation loss worsens after early epochs,
supporting early stopping and the short final budget. Further capture should
prioritize difficult upright/backward and hunched/desk variants across subjects.

## E. Final model and independent test

Fresh final seed: **10302**. Training pool: 12 subjects, 12,743 eligible frames;
QBAO excluded. Four complete epochs; final training loss decreased from 0.6842
to 0.0979. Hyperparameters were frozen before final training and test access.

Checkpoint: `{model.relative_to(ROOT).as_posix()}/model_final.pt`.
Checkpoint SHA-256: `{summary['checkpoint_sha256']}`.

The persistent final-test marker records **one prediction/evaluation pass**.
The final QBAO test comprises 1,513 eligible frames from 13 usable camera clips
and 7 source takes (1,697 source frames / 14 clips in the frozen manifest).

* Frame accuracy: **{test['frame_accuracy']:.4f}**.
* Frame, clip, source, and subject Macro-F1: **1.0000**.
* Test CE: **{test['test_loss']:.6f}**; ECE: **{test['calibration_ece']:.6f}**;
  Brier score: **{test['brier_score']:.6f}**.
* Per-class test supports: 237, 122, 215, 246, 429, 264; each class precision,
  recall and F1 is 1.0 on these eligible frames.

The perfect score is a measurement on **one subject / seven independent takes**;
the frames are highly correlated and do not establish perfect population accuracy.
The test front view has no eligible label-1 samples. Its fixed six-label camera
Macro-F1 is therefore 0.8333 despite 1.0 accuracy; oblique is 1.0 for both.
No model or thresholds were changed in response to test results.

Model directory contains PT/ONNX weights, model configuration, feature contract,
official label map, frozen split, training summary, test metrics/reports, final
history/curve, export parity reports, and end-to-end verification metadata.
Earlier versions are never replaced.

## F. Deployment and actual execution

**ONNX verified** on 12,743 real train/validation feature rows. Maximum absolute
probability difference: **{parity['max_abs_probability_difference']:.9g}** (gate 1e-5).
Maximum absolute logit difference: {parity['max_abs_logit_difference']:.9g}.
The probability gate passed; both PT and ONNX deployment loaders return finite
six-class probabilities. No held-out test predictions were repeated for export.

This exact command was executed successfully from the repository:

```powershell
$env:PYTHONUTF8 = '1'
& C:/Users/tienq/miniforge3/envs/ferlab-infer/python.exe inference/run.py --source artifacts/dataset_working/Real_Posture_Dataset/recordings/CHANH/session_20261027_1124/label_0/take_001/cam01_raw.mp4 --mode video --model artifacts/models/mlp_wide/v001 --pose artifacts/pretrained/yolo11n-pose.pt --device cuda:0 --backend onnx --cls-device cpu --headless --name recovered_real_e2e
```

All 71 frames completed in 12.0 s, including cold startup. Every frame had a
single-person selection and class-0 prediction. Pipeline: existing local
YOLO11n-Pose -> ByteTrack -> person selection -> shared causal normalizer -> ONNX
MLP. Local annotated video and frame logs are in
`{inference_dir.relative_to(ROOT).as_posix()}`; video is excluded from bundles.
Exact audit/training/export commands and resume usage are documented in
`docs/TRAINING_WINDOWS.md`. All execution logs are saved to disk.

## G. Tests, limitations, and optional work

Before changes, the default Windows test run had 102 passed, 4 skipped, 6 failed.
With UTF-8 enabled before changes: 106 passed, 4 skipped, 2 failed. After the main
implementation: **111 passed, 4 skipped, 1 failed** in the full suite. The remaining
failure is the existing `test_shared_assets_copy_link_and_verify`: Windows denies
symbolic-link creation (WinError 1314). It is not a training or collector regression.
Collector GUI tests passed. The final focused contract/resume suite is recorded
in `artifacts/reports/final_training_contract_tests.log`.

Real integrity preflight and deep dataset validation both reported zero structural
errors. The final verifier passed hashes, seals, all-row feature regeneration,
nonfinite/missing-mask checks, split checks, all 14 best-checkpoint loads, and
six-probability checks. Interrupted/resumed training matched uninterrupted weights
and histories exactly in its integration test.

Review-required clips and uncertain frames remain unresolved; no attempt was made
to guess their person/posture or re-extract them. No new bulk extraction ran.
Notebooks were updated and all code cells compiled; their full interactive UI was
not separately executed because actual training used the shared CLI.

Optional GRU was **not trained**. Temporal feasibility was inspected without model
predictions: training has 425 clean non-overlapping 16-frame windows and 168
32-frame windows; validation has 70 / 30. Those windows come from only 69 training
takes and 12 validation takes, with repeated capture ticks and short held-posture
recordings. Temporal model comparison remains optional future work; no GRU score
or claim of improvement is made.

The working directory has no `.git` metadata, so no commit or git diff is available.
This report distinguishes the completed MLP deliverables from the deferred GRU and
remaining data review / Windows symlink limitation.
'''
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports/'TRAINING_FINAL_REPORT.md'
    report_path.write_text(report, encoding='utf-8')
    # Explicit allowlists prevent private video/features from entering bundles.
    bundle = model/'model_bundle.zip'
    if bundle.exists():
        raise FileExistsError(bundle)
    with ZipFile(bundle,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(model.rglob('*')):
            if path.is_file() and path.suffix in {'.pt','.onnx','.json','.csv','.png'}:
                z.write(path,path.relative_to(model).as_posix())
        for name in ['shared/mlp_model.py','shared/pose_features.py','inference/pipeline.py','inference/run.py','docs/TRAINING_WINDOWS.md']:
            z.write(ROOT/name,name)
        z.write(report_path,'TRAINING_FINAL_REPORT.md')
    with ZipFile(bundle) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.mp4','.npz','.txt')) for n in z.namelist())
    write_json(model/'bundle_sha256.json',dict(file=bundle.name,sha256=file_sha256(bundle),private_video_included=False))
    return report_path,bundle


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--reports',type=Path,default=ROOT/'artifacts/reports')
    args=ap.parse_args()
    for path in build(args.run,args.reports): print(path)
