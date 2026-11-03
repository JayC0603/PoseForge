#!/usr/bin/env python3
"""Read-only integrity/feature checks after training; never re-evaluates test model scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from shared.dataset_pipeline import file_sha256
from shared.real_training import (TrainingData, pose_path, read_json, stream_features,
                                  uncertain_selection, verify_seal, write_json)
from shared.mlp_model import PostureMLP


def verify(run, data_root):
    run, data_root = Path(run).resolve(), Path(data_root).resolve()
    inventory = read_json(run / "snapshot/source_inventory.json")
    changed = [rel for rel, value in inventory.items()
               if not (data_root / rel).is_file() or file_sha256(data_root / rel) != value["sha256"]]
    if changed:
        raise RuntimeError(f"Original source changed: {changed}")
    verify_seal(run / "snapshot"); verify_seal(run / "processed")
    split = pd.read_csv(run / "split_manifest.csv")
    assert split.groupby("person_id").split.nunique().eq(1).all()
    assert split.groupby("source_group_id").split.nunique().eq(1).all()
    checked_frames = 0
    sequence_rows = []
    catalog = pd.read_csv(run / "catalog.csv")
    for row in catalog.to_dict("records"):
        items = [json.loads(line) for line in pose_path(run / "snapshot", row).read_text(encoding="utf-8").splitlines()]
        selected = [dict(item, keypoints=None) if uncertain_selection(item) else item for item in items]
        regenerated = stream_features(selected)
        with np.load(run / "processed" / f'{row["clip_id"]}.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(regenerated, data["x"])
            assert np.isfinite(data["x"]).all()
            assert data["x"].shape == (int(row["frames"]), 13, 4)
            assert np.all(data["x"][:,:,:2][data["x"][:,:,3] == 0] == 0)
            assert np.all(data["x"][:,:,2][data["x"][:,:,3] == 0] == 0)
            assert np.isin(data["x"][:,:,3], [0,1]).all()
            eligible, ticks = data["eligible"], data["capture_ns"]
        # Assess non-overlapping temporal windows only, without model predictions.
        for length in (16,32):
            clean = 0
            for start in range(0, len(items)-length+1, length):
                end = start+length
                delta = np.diff(ticks[start:end])
                clean += int(eligible[start:end].all() and (delta > 0).all()
                             and (delta <= 250_000_000).all())
            sequence_rows.append(dict(clip_id=row["clip_id"], person=row["person_id"],
                                      split=row["split"], label=int(row["label"]), camera=row["camera_id"],
                                      sequence_length=length, valid_nonoverlap_windows=clean))
        checked_frames += len(items)
    # Load every selected best checkpoint, using train/val observations only.
    training = TrainingData(run, ["train", "val"], "cpu")
    sample = torch.from_numpy(training.x[training.ids(training.clips)[:32]])
    loaded = []
    with torch.inference_mode():
        for file in sorted((run / "experiments").glob("*/best.pt")):
            checkpoint = torch.load(file, weights_only=True, map_location="cpu")
            model = PostureMLP(checkpoint["model_name"], checkpoint["model_config"]["dropout"]).eval()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            p = model(sample).softmax(1)
            assert p.shape == (len(sample),6) and torch.isfinite(p).all()
            np.testing.assert_allclose(p.sum(1).numpy(), 1, atol=1e-6)
            loaded.append(file.parent.name)
    seq = pd.DataFrame(sequence_rows)
    seq.to_csv(run / "temporal_feasibility.csv", index=False)
    temporal = seq.groupby(["split", "sequence_length"]).valid_nonoverlap_windows.sum().reset_index().to_dict("records")
    report = dict(source_hashes_unchanged=True, source_files_checked=len(inventory),
                  snapshot_and_features_sealed=True, regenerated_feature_frames=checked_frames,
                  feature_dim=52, finite_inputs=True, missing_masks_preserved=True,
                  subject_overlap=False, source_take_overlap=False,
                  best_checkpoints_loaded=loaded, six_class_probability_checks=True,
                  temporal_feasibility=temporal,
                  temporal_status="GRU deferred: windows are possible, but only 88 independent takes and irregular capture ticks; no GRU performance claimed.",
                  test_model_predictions_repeated=False)
    write_json(run / "quality_gates.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/dataset_working/Real_Posture_Dataset")
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.data), indent=2))
