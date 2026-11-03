"""Leakage, causal deployment parity, and immutable real training contracts."""
import json

import numpy as np
import pytest

from shared.real_training import (TrainingData, seal, stream_features, uncertain_selection,
                                  verify_seal, write_json)
from inference import pipeline


def item(kp, track=1):
    return dict(keypoints=kp, width=640, height=480, track_id=track,
                selected_track_id=track, status="ok", person_count=1,
                selection_state="selected")


def points(offset=0):
    return [[float(200+j*4+offset), float(150+j*7), .9] for j in range(17)]


def test_stream_matches_deployment_across_gaps_and_track_changes(monkeypatch):
    rows = [item(points()), item(points(20))] + [item(None)]*10 + [item(points(90)), item(points(110), 2)]

    class Replay:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def track_frame(self, frame):
            r = rows[self.index]; self.index += 1
            return [] if r["keypoints"] is None else [dict(keypoints=r["keypoints"], track_id=r["track_id"], bbox=[0,0,640,480])]

        def close(self):
            pass

    class Classifier:
        conf_thresh, ema_alpha = .35, .25
        labels = [str(i) for i in range(6)]
        def __init__(self):
            self.seen = []
        def predict(self, x):
            self.seen.append(x.copy())
            return np.ones(6)/6

    monkeypatch.setattr(pipeline, "PoseRunner", Replay)
    classifier = Classifier()
    deploy = pipeline.PosePipeline("unused", classifier)
    for r in rows:
        deploy.process(np.zeros((480,640,3), np.uint8))
    generated = stream_features(rows)
    expected = generated[[i for i,r in enumerate(rows) if r["keypoints"] is not None]]
    np.testing.assert_array_equal(expected, np.stack(classifier.seen))
    np.testing.assert_array_equal(generated[:2], stream_features(rows[:2]))
    assert np.isfinite(generated).all() and np.all(generated[2:12] == 0)


def test_ambiguous_selection_cannot_enter_supervision():
    assert uncertain_selection(item(None))
    multi = item(points()); multi["person_count"] = 2; multi["persons"] = []
    assert uncertain_selection(multi)
    assert not uncertain_selection(item(points()))


def test_sealed_snapshot_detects_tampering(tmp_path):
    path = tmp_path / "provenance.json"
    write_json(path, {"dataset_id":"real"})
    seal(tmp_path); verify_seal(tmp_path)
    path.chmod(0o666)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Immutable"):
        verify_seal(tmp_path)


def test_training_refuses_changed_frozen_split(tmp_path):
    path = tmp_path / "split_manifest.json"
    write_json(path, {"test_people":["held_out"]})
    write_json(tmp_path / "frozen_manifest_checksums.json", {"split_manifest.json":"wrong_hash"})
    with pytest.raises(RuntimeError, match="Frozen"):
        TrainingData(tmp_path, ["train","val"], "cpu")


def test_interrupted_training_resumes_same_optimizer_and_rng(tmp_path, monkeypatch):
    import pandas as pd
    import torch
    from shared import real_training as rt
    torch.set_num_threads(2)
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = []
    rng = np.random.default_rng(17)
    for split in ("train", "val"):
        for label in range(6):
            cid = f"{split}_{label}"
            x = rng.normal(size=(8,13,4)).astype(np.float32)
            x[:,:,0] += label
            x[:,:,2:] = 1
            np.savez_compressed(processed / f"{cid}.npz", x=x, eligible=np.ones(8,bool))
            rows.append(dict(clip_id=cid, person_id=split, label=label, camera_id="cam01",
                             camera_view="front", source_group_id=cid, split=split))
    pd.DataFrame(rows).to_csv(tmp_path / "catalog.csv", index=False)
    write_json(tmp_path / "frozen_manifest_checksums.json", {})
    rt.seal(processed)
    cfg = {**rt.DEFAULT, "model":"mlp_small", "epochs":3, "patience":5}
    reference = rt.run_candidate(tmp_path, cfg, "cpu", "reference")
    original = rt.train_epoch
    def interrupted(*args, **kwargs):
        if args[-1] == 3:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(rt, "train_epoch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        rt.run_candidate(tmp_path, cfg, "cpu", "interrupted")
    monkeypatch.setattr(rt, "train_epoch", original)
    resumed = rt.run_candidate(tmp_path, cfg, "cpu", "interrupted", resume=True)
    assert reference["metrics"] == resumed["metrics"]
    a = torch.load(tmp_path / "experiments/reference/last.pt", weights_only=False)
    b = torch.load(tmp_path / "experiments/interrupted/last.pt", weights_only=False)
    assert a["history"] == b["history"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
