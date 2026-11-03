"""Reproducible Windows training on frozen real recordings (no source writes).

The snapshot contains pose/timestamp/metadata copies and video hashes, never videos.
Validation and final test have separate entry points; tuning never loads test arrays.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shared.dataset_pipeline import COCO13, file_sha256
from shared.pose_features import CausalPoseNormalizer, FEATURE_VERSION
from shared.pose_training import (augment_pose13_torch, frozen_split,
                                  group_sample_epoch, source_group_id, usable_mask)

LOG = logging.getLogger("poseforge.training")
LABELS = list(range(6))
DEFAULT = dict(seed=302, model="mlp_base", dropout=.15, lr=.001,
               weight_decay=.001, batch_size=128, epochs=100, patience=12,
               class_weight="none", label_smoothing=0., quota=0, balance="none")
AUGMENTATION = dict(flip_prob=.5, jitter_prob=.5, jitter_sigma=.01,
                    confidence_perturb_prob=.2, keypoint_dropout_prob=.1,
                    occlusion_prob=.1, max_dropout_joints=2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def version_dir(parent, prefix="v"):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, 100000):
        path = parent / f"{prefix}{i:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("Version namespace exhausted")


def inside(root, value):
    rel = Path(str(value).replace("\\", "/"))
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe dataset member {value!r}")
    return path


def pose_path(root, row):
    return inside(root, row["video_path"]).parent / "keypoints" / f'{row["camera_id"]}_yolo17_raw.txt'


def uncertain_selection(item):
    """Never guess a person. Multi-box selection needs geometric duplicate proof.

    Existing explicit single-person observations are retained, including partial
    joints. No attempt to recover an unassigned/missing selection is made here.
    """
    if item.get("keypoints") is None or item.get("status") != "ok":
        return True
    if item.get("selection_state") in {"selection_required", "missing", "ambiguous_candidates"}:
        return True
    if int(item.get("person_count", 1)) <= 1:
        return False
    from collector.posture_collector.single_person import SinglePersonSelector
    selector = SinglePersonSelector()
    person, reason = selector.select(item.get("persons", []))
    if reason != "duplicate_boxes" or person is None:
        return True
    return not np.allclose(np.asarray(person["keypoints"]), np.asarray(item["keypoints"]), atol=1e-5, rtol=0)


def stream_features(items):
    """Mirror PosePipeline's causal state boundaries on stored selected poses.

    Includes existing scale EMA only; no extra keypoint smoothing. Retains all
    missing frames. Track IDs are local to each clip, never compared across clips.
    """
    norm = CausalPoseNormalizer()
    last_track, missing = None, 0
    result = []
    for item in items:
        kp = item.get("keypoints")
        track = item.get("selected_track_id", item.get("track_id"))
        w, h = int(item["width"]), int(item["height"])
        if kp is None:
            x = norm.transform(None, w, h)[0]
            missing += 1
            if missing > 8:
                norm.reset()
                last_track = None
        else:
            if track is not None and last_track is not None and track != last_track:
                norm.reset()
            if track is not None:
                last_track = track
            missing = 0
            x = norm.transform(kp, w, h)[0]
        result.append(x)
    return np.asarray(result, dtype=np.float32)


def audit(root, report_dir, recovery_manifest=None):
    import cv2
    root, report_dir = Path(root).resolve(), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(root / "labels.csv", encoding="utf-8-sig")
    if catalog.clip_id.duplicated().any() or set(catalog.label) != set(LABELS):
        raise ValueError("Duplicate clip IDs or incorrect six-class mapping")
    if (root / "curation_manifest.json").exists():
        lineage = read_json(root / "curation_manifest.json")
        by_clip = {r["clip_id"]: r for r in lineage["recordings"]}
        if set(by_clip) != set(catalog.clip_id):
            raise ValueError("Existing clean lineage does not match labels.csv")
    else:
        by_clip = {}
    edits_path = root.parent / "curation.json"
    edits = read_json(edits_path).get("takes", {}) if edits_path.exists() else {}
    files = {root / "labels.csv", *list((root / "config").glob("*.json"))}
    if (root / "curation_manifest.json").exists():
        files.add(root / "curation_manifest.json")
    for row in catalog.to_dict("records"):
        video = inside(root, row["video_path"])
        files.update([video, inside(root, row["metadata_path"]), pose_path(root, row),
                      video.parent / f'{row["camera_id"]}_timestamps.csv'])
        files.update(pose_path(root, row).parent.glob(f'{row["camera_id"]}*.summary.json'))
    inventory = {}
    for file in sorted(files):
        if not file.is_file():
            continue
        stat = file.stat()
        if time.time() - stat.st_mtime < 60:
            raise RuntimeError(f"Source recently modified; wait until recording/extraction is idle: {file.name}")
        digest = file_sha256(file)
        if (stat.st_size, stat.st_mtime_ns) != (file.stat().st_size, file.stat().st_mtime_ns):
            raise RuntimeError("Source changed while hashing")
        inventory[file.relative_to(root).as_posix()] = dict(sha256=digest, bytes=stat.st_size,
                                                         mtime_ns=stat.st_mtime_ns)
    errors, clips, frame_rows = [], [], []
    states, selection_states, joint_valid = Counter(), Counter(), np.zeros(13, dtype=np.int64)
    ranges, duplicate_frames = defaultdict(list), defaultdict(list)
    for row in catalog.to_dict("records"):
        cid = row["clip_id"]
        video = inside(root, row["video_path"])
        meta = read_json(inside(root, row["metadata_path"]))
        kp_file = pose_path(root, row)
        if (meta.get("person_id") != row["person_id"] or int(meta["label"]) != int(row["label"])
                or meta["take_id"] != row["take_id"]):
            errors.append(f"{cid}: metadata identity mismatch")
        items = [json.loads(line) for line in kp_file.read_text(encoding="utf-8").splitlines()] if kp_file.exists() else []
        times_path = video.parent / f'{row["camera_id"]}_timestamps.csv'
        times = pd.read_csv(times_path)
        ticks = times.host_recv_monotonic_ns.to_numpy(dtype=np.int64)
        delta = np.diff(ticks)
        if len(items) != int(row["frames"]) or len(times) != int(row["frames"]):
            errors.append(f"{cid}: pose/timestamp/frame counts mismatch")
        if times.writer_frame_index.tolist() != list(range(len(times))):
            errors.append(f"{cid}: non-contiguous timestamp writer indices")
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        decoded, last_digest, adjacent_repeat = 0, None, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                digest = hashlib.sha256(frame.tobytes()).hexdigest()
                adjacent_repeat += int(digest == last_digest)
                last_digest = digest
                # Full decoded-frame hashes detect duplicate recordings and frames.
                duplicate_frames[digest].append((cid, decoded))
                decoded += 1
        finally:
            cap.release()
        if decoded != int(row["frames"]):
            errors.append(f"{cid}: decoded {decoded} != manifest {row['frames']}")
        source = by_clip.get(cid, {})
        source_take = source.get("source_take_id", row["take_id"])
        group = source_group_id(row["person_id"], row["session_id"], row["label"], source_take)
        start = int(source.get("source_frame_start", 0))
        end = int(source.get("source_frame_end_exclusive", len(items)))
        if end - start != len(items) or start < 0:
            errors.append(f"{cid}: invalid source range")
        ranges[(group, row["camera_id"])].append((start, end, cid))
        take_rel = video.parent.relative_to(root / "recordings").as_posix()
        edit = edits.get(take_rel, {})
        excluded = edit.get("exclude_take", False) or row["camera_id"] in edit.get("exclude_cameras", [])
        camera_status = meta["cameras"][row["camera_id"]].get("status")
        camera_bad = camera_status not in {"ok", "complete"} or row["status"] != "ok"
        pose_count, eligible_count, ambiguous_count, full_count, switches = 0, 0, 0, 0, 0
        last_track = None
        for i, item in enumerate(items):
            states[str(item.get("status"))] += 1
            selection_states[str(item.get("selection_state"))] += 1
            if int(item["frame_index"]) != i or item.get("backend") != "yolo":
                errors.append(f"{cid}: invalid frame index/backend at {i}")
            kp = item.get("keypoints")
            nvalid = 0
            if kp is not None:
                raw = np.asarray(kp, dtype=np.float32)
                if raw.shape != (17, 3) or not np.isfinite(raw).all():
                    errors.append(f"{cid}: malformed/nonfinite pose at {i}")
                    continue
                pose_count += 1
                valid = ((raw[:13, 2] >= .35) & (raw[:13, :2] >= 0).all(1)
                         & (raw[:13, :2] <= (item["width"], item["height"])).all(1))
                nvalid = int(valid.sum())
                joint_valid += valid
                full_count += int(nvalid == 13)
                track = item.get("selected_track_id", item.get("track_id"))
                switches += int(track is not None and last_track is not None and track != last_track)
                if track is not None:
                    last_track = track
            uncertain = uncertain_selection(item)
            ambiguous_count += int(kp is not None and uncertain)
            kept = not excluded
            if edit.get("keep_ranges_ns") is not None:
                kept &= any(a <= int(ticks[i]) < b for a, b in edit["keep_ranges_ns"])
            eligible = kept and not camera_bad and not uncertain and nvalid >= 4
            eligible_count += int(eligible)
            frame_rows.append(dict(clip_id=cid, frame_index=i, source_frame_index=start+i,
                                   person_id=row["person_id"], label=int(row["label"]),
                                   camera_id=row["camera_id"], valid_joints=nvalid,
                                   selected=kp is not None, review_required=uncertain,
                                   eligible=bool(eligible), capture_ns=int(ticks[i])))
        span = float((ticks[-1]-ticks[0])/1e9) if len(ticks) > 1 else 0.
        positive = delta[delta > 0]
        median = float(np.median(positive)) if len(positive) else 0.
        clips.append(dict(**row, source_take_id=source_take, source_group_id=group,
                          source_frame_start=start, source_frame_end_exclusive=end,
                          pose17_frames=pose_count, usable_frames=eligible_count,
                          full13_frames=full_count, no_selected_frames=len(items)-pose_count,
                          uncertain_selected_frames=ambiguous_count, track_switches=switches,
                          raw_available=kp_file.exists(), camera_status=camera_status,
                          review_required=bool(ambiguous_count or pose_count == 0 or camera_bad),
                          decoded_frames=decoded, encoded_fps=fps,
                          encoded_duration_s=decoded/fps if fps else 0., capture_span_s=span,
                          duplicate_timestamps=int((delta == 0).sum()),
                          backward_timestamps=int((delta < 0).sum()),
                          large_timestamp_gaps=int((delta > max(250e6, median*5)).sum()),
                          max_timestamp_gap_ms=float(delta.max()/1e6) if len(delta) else 0.,
                          adjacent_repeated_video_frames=adjacent_repeat))
        if len(clips) % 20 == 0:
            LOG.info("Audited %d/%d clips", len(clips), len(catalog))
    for key, pieces in ranges.items():
        for a, b in zip(sorted(pieces), sorted(pieces)[1:]):
            if a[1] > b[0]:
                errors.append(f"Overlapping source ranges: {key} {a[2]} {b[2]}")
    clip_df, frames = pd.DataFrame(clips), pd.DataFrame(frame_rows)
    duplicates = [dict(sha256=digest, occurrences=locations) for digest, locations in duplicate_frames.items()
                  if len({cid for cid, _ in locations}) > 1]
    # Exclude exact cross-clip duplicate observations conservatively, no leakage.
    duplicate_locations = {tuple(loc) for group in duplicates for loc in group["occurrences"]}
    if duplicate_locations:
        frames.loc[[tuple(v) in duplicate_locations for v in frames[["clip_id", "frame_index"]].to_numpy()], "eligible"] = False
        clipped_counts = frames.groupby("clip_id").eligible.sum()
        clip_df["usable_frames"] = clip_df.clip_id.map(clipped_counts)
    recovery = []
    if recovery_manifest and Path(recovery_manifest).exists():
        for result in read_json(recovery_manifest)["results"]:
            target = result["target"].split(" ", 1)[1].split("/")
            person, session, label, take, cam = target
            match = clip_df[(clip_df.person_id == person) & (clip_df.session_id == session)
                            & (clip_df.label == int(label.split("_")[1]))
                            & (clip_df.camera_id == cam) & clip_df.video_path.str.replace("\\", "/", regex=False).str.contains(f"/{take}/", regex=False)]
            recovery.append(dict(target=result["target"], matched_clips=len(match),
                                 pose_frames=int(match.pose17_frames.sum()),
                                 recognized=bool(len(match) == 1 and match.pose17_frames.sum() > 0)))
    canonical = {k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in inventory.items()}
    data_id = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    summary = dict(dataset_id=data_id, dataset_root=str(root), created_utc=datetime.now(timezone.utc).isoformat(),
                   subjects=sorted(catalog.person_id.unique()), subject_count=int(catalog.person_id.nunique()),
                   clips=len(catalog), source_takes=int(clip_df.source_group_id.nunique()),
                   frames=int(catalog.frames.sum()), decoded_frames=int(clip_df.decoded_frames.sum()),
                   label_clips={str(k): int(v) for k, v in catalog.label.value_counts().sort_index().items()},
                   label_frames={str(k): int(v) for k, v in catalog.groupby("label").frames.sum().items()},
                   camera_clips=catalog.camera_id.value_counts().to_dict(),
                   camera_frames={k:int(v) for k,v in catalog.groupby("camera_id").frames.sum().items()},
                   encoded_duration_s=float(clip_df.encoded_duration_s.sum()),
                   capture_duration_s=float(clip_df.capture_span_s.sum()),
                   raw_files_present=int(clip_df.raw_available.sum()),
                   empty_pose_clips=clip_df.loc[clip_df.pose17_frames == 0, "clip_id"].tolist(),
                   pose_frames=int(clip_df.pose17_frames.sum()), eligible_frames=int(frames.eligible.sum()),
                   no_selected_frames=int(clip_df.no_selected_frames.sum()),
                   uncertain_selected_frames=int(clip_df.uncertain_selected_frames.sum()),
                   joint_valid_counts=dict(zip(COCO13, map(int, joint_valid))),
                   joint_missing_counts=dict(zip(COCO13, map(int, len(frames)-joint_valid))),
                   full13_frames=int(clip_df.full13_frames.sum()),
                   raw_status=dict(states), selection_states=dict(selection_states),
                   track_switches=int(clip_df.track_switches.sum()),
                   duplicate_timestamps=int(clip_df.duplicate_timestamps.sum()),
                   backward_timestamps=int(clip_df.backward_timestamps.sum()),
                   large_timestamp_gaps=int(clip_df.large_timestamp_gaps.sum()),
                   cross_clip_duplicate_frame_hashes=len(duplicates),
                   repeated_adjacent_frames=int(clip_df.adjacent_repeated_video_frames.sum()),
                   recovery=recovery, recovery_recognized=sum(r["recognized"] for r in recovery),
                   existing_clean_lineage=bool(by_clip), curation_edits=edits,
                   quality_policy="Keep partial joints; exclude uncertain/no selection, <4 joints, camera errors, curated exclusions and exact cross-clip duplicates. No relabeling.",
                   errors=errors)
    # Detect writes during audit, including videos. Before prepare, hashes are checked again.
    for rel, info in inventory.items():
        stat = (root / rel).stat()
        if (stat.st_size, stat.st_mtime_ns) != (info["bytes"], info["mtime_ns"]):
            raise RuntimeError(f"Source modified during audit: {rel}")
    write_json(report_dir / "source_inventory.json", inventory)
    write_json(report_dir / "duplicate_frames.json", duplicates)
    clip_df.to_csv(report_dir / "clip_quality.csv", index=False)
    frames.to_csv(report_dir / "frame_quality.csv.gz", index=False, compression="gzip")
    coverage = frames.groupby(["person_id", "label"]).agg(frames=("eligible", "size"), eligible=("eligible", "sum"), mean_valid_joints=("valid_joints", "mean")).reset_index()
    coverage.to_csv(report_dir / "subject_coverage.csv", index=False)
    write_json(report_dir / "training_preflight.json", summary)
    md = f"# Recovered real dataset preflight\n\nDataset ID: `{data_id}`\n\n{summary['subject_count']} subjects, {summary['source_takes']} source takes, {len(catalog)} camera clips, {summary['frames']} decoded frames.\n\n"
    md += f"Pose observations: {summary['pose_frames']}; eligible supervised frames: {summary['eligible_frames']}; no selected person: {summary['no_selected_frames']}; uncertain selected person: {summary['uncertain_selected_frames']}.\n\n"
    md += f"Recovered files recognized: {summary['recovery_recognized']}/{len(recovery)}. Empty pose files: {len(summary['empty_pose_clips'])}.\n\n"
    md += f"Timestamp repeats: {summary['duplicate_timestamps']}; backward ticks: {summary['backward_timestamps']}; large gaps: {summary['large_timestamp_gaps']}. Source order is preserved.\n\n"
    md += f"Cross-clip duplicate frame hashes: {len(duplicates)}. Structural errors: {len(errors)}.\n\n{summary['quality_policy']}\n\n"
    md += "Class frames: " + json.dumps(summary["label_frames"]) + "\n\nSubjects: " + ", ".join(summary["subjects"]) + "\n\n"
    md += "See clip_quality.csv and subject_coverage.csv for per-clip and per-subject evidence. Take-level needs_review alone is not a camera failure; camera statuses are checked separately. This automated export is not a manual posture approval.\n"
    (report_dir / "training_preflight.md").write_text(md, encoding="utf-8")
    if errors:
        raise ValueError(f"Audit found structural errors: {errors[:5]}")
    return summary


def seal(folder):
    """Content seal; Windows read-only attributes add protection against accidents."""
    checks = {f.relative_to(folder).as_posix(): file_sha256(f)
              for f in sorted(folder.rglob("*")) if f.is_file() and f.name != "checksums.json"}
    write_json(folder / "checksums.json", checks, exclusive=True)
    for file in folder.rglob("*"):
        if file.is_file():
            file.chmod(0o444)


def verify_seal(folder):
    for rel, digest in read_json(folder / "checksums.json").items():
        if file_sha256(folder / rel) != digest:
            raise RuntimeError(f"Immutable artifact changed: {folder / rel}")


def prepare(root, report_dir, output_root, validation=None, test=None, seed=302):
    root, report_dir = Path(root).resolve(), Path(report_dir)
    report = read_json(report_dir / "training_preflight.json")
    inventory = read_json(report_dir / "source_inventory.json")
    catalog = pd.read_csv(report_dir / "clip_quality.csv")
    quality = pd.read_csv(report_dir / "frame_quality.csv.gz")
    # Hash comparison, rather than file existence, gates creating an immutable copy.
    for rel, info in inventory.items():
        if file_sha256(root / rel) != info["sha256"]:
            raise RuntimeError("Source changed since audit; create a new audit: " + rel)
    coverage = quality.groupby(["person_id", "label"]).eligible.sum().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    candidates = sorted(coverage.index[(coverage >= 16).all(axis=1)])
    npeople = catalog.person_id.nunique()
    nval = max(1, round(npeople * 2 / 13))
    ntest = max(1, round(npeople / 13))
    shuffled = np.random.default_rng(seed).permutation(candidates).tolist()
    test = list(test) if test else shuffled[:ntest]
    validation = list(validation) if validation else [p for p in shuffled if p not in test][:nval]
    if any(p not in candidates for p in test + validation):
        raise ValueError("Evaluation subjects need >=16 eligible frames in each of six classes")
    split, info = frozen_split(catalog, validation, test, seed)
    info.update(dataset_id=report["dataset_id"], evaluation_selection="seeded permutation of subjects with >=16 eligible frames/class; quality only, no model scores", eligible_evaluation_subjects=candidates)
    split = split.merge(catalog[["clip_id", "source_group_id", "source_frame_start", "source_frame_end_exclusive"]], on="clip_id", validate="one_to_one")
    if (split.groupby("person_id").split.nunique() != 1).any() or (split.groupby("source_group_id").split.nunique() != 1).any():
        raise ValueError("Subject/source leakage")
    run = version_dir(Path(output_root), "real_v")
    snapshot, processed = run / "snapshot", run / "processed"
    snapshot.mkdir(); processed.mkdir()
    for rel, record in inventory.items():
        src = root / rel
        if src.suffix.lower() == ".mp4":
            continue
        dst = snapshot / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if file_sha256(dst) != record["sha256"] or file_sha256(src) != record["sha256"]:
            raise RuntimeError("Source changed while copying snapshot")
    if (root.parent / "curation.json").exists():
        if read_json(root.parent / "curation.json").get("takes", {}) != report["curation_edits"]:
            raise RuntimeError("Curation edits changed since audit")
    write_json(snapshot / "source_inventory.json", inventory, exclusive=True)
    lineage = dict(version=1, kind="automated_non_destructive_training_snapshot", dataset_id=report["dataset_id"],
                   source_dataset=str(root), human_approved=False, quality_policy=report["quality_policy"],
                   curation_edits=report["curation_edits"], recordings=[])
    split_map = split.set_index("clip_id").split.to_dict()
    for row in catalog.to_dict("records"):
        items = [json.loads(line) for line in pose_path(snapshot, row).read_text(encoding="utf-8").splitlines()]
        selected_items = [dict(item, keypoints=None) if uncertain_selection(item) else item for item in items]
        x = stream_features(selected_items)
        if x.shape != (int(row["frames"]), 13, 4) or not np.isfinite(x).all():
            raise ValueError("Invalid 52-feature data")
        q = quality[quality.clip_id == row["clip_id"]].sort_values("frame_index")
        eligible = q.eligible.to_numpy(dtype=bool) & usable_mask(x)
        # Prefix invariance verifies no future dependence.
        np.testing.assert_array_equal(x[:min(12, len(x))], stream_features(selected_items[:min(12, len(x))]))
        np.savez_compressed(processed / f'{row["clip_id"]}.npz', x=x, eligible=eligible,
                            frame_index=q.frame_index.to_numpy(np.int64),
                            source_frame_index=q.source_frame_index.to_numpy(np.int64),
                            capture_ns=q.capture_ns.to_numpy(np.int64))
        lineage["recordings"].append(dict(clip_id=row["clip_id"], take_id=row["take_id"],
              source_take_id=row["source_take_id"], source_clip_id=row["clip_id"], camera_id=row["camera_id"],
              source_frame_start=int(row["source_frame_start"]), source_frame_end_exclusive=int(row["source_frame_end_exclusive"]),
              segment=1, frames=int(row["frames"]), eligible_frames=int(eligible.sum())))
    write_json(snapshot / "training_curation_manifest.json", lineage, exclusive=True)
    contract = dict(version=FEATURE_VERSION, extractor="yolo11n-pose.pt", joints=COCO13,
                    channels=["x_norm", "y_norm", "confidence", "valid_mask"],
                    shape_per_frame=[13, 4], flatten_dim=52, confidence_threshold=.35, ema_alpha=.25,
                    normalization="shared.CausalPoseNormalizer; state resets on clip, changed selected track, >8 missing frames",
                    missing_policy="uncertain selections enter causal stream as absent; zero invalid xy/conf/mask; preserve all rows; supervised excludes <4 joints",
                    dataset_id=report["dataset_id"])
    write_json(run / "feature_contract.json", contract, exclusive=True)
    write_json(run / "split_manifest.json", info, exclusive=True)
    split.to_csv(run / "split_manifest.csv", index=False)
    catalog.assign(split=catalog.clip_id.map(split_map)).to_csv(run / "catalog.csv", index=False)
    pd.read_csv(report_dir / "subject_coverage.csv").assign(split=lambda df: df.person_id.map(split.drop_duplicates("person_id").set_index("person_id").split)).to_csv(run / "subject_coverage.csv", index=False)
    write_json(run / "dataset_summary.json", report, exclusive=True)
    seal(snapshot); seal(processed)
    # Seal split + contract independently from mutable experiment outputs.
    write_json(run / "frozen_manifest_checksums.json", {n: file_sha256(run / n) for n in
               ["feature_contract.json", "split_manifest.json", "split_manifest.csv", "catalog.csv"]}, exclusive=True)
    LOG.info("Prepared %s; frozen split %s", run, info)
    return run


class TrainingData:
    def __init__(self, run, splits, device):
        import torch
        self.run = Path(run)
        for name, digest in read_json(self.run / "frozen_manifest_checksums.json").items():
            if file_sha256(self.run / name) != digest:
                raise RuntimeError("Frozen split/contract changed")
        catalog = pd.read_csv(self.run / "catalog.csv")
        catalog = catalog[catalog.split.isin(splits)]
        xs, ys, valid, self.clips = [], [], [], []
        offset = 0
        checks = read_json(self.run / "processed/checksums.json")
        for r in catalog.to_dict("records"):
            file = self.run / "processed" / f'{r["clip_id"]}.npz'
            if file_sha256(file) != checks[file.name]:
                raise RuntimeError("Processed feature hash mismatch")
            with np.load(file, allow_pickle=False) as data:
                x, eligible = data["x"].reshape(-1, 52), data["eligible"]
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite training input")
            xs.append(x); valid.append(eligible); ys.append(np.full(len(x), r["label"], np.int64))
            self.clips.append(dict(start=offset, end=offset+len(x), label=int(r["label"]),
                clip_id=r["clip_id"], person=r["person_id"], camera=r["camera_id"],
                camera_view=r["camera_view"], source_group_id=r["source_group_id"], split=r["split"]))
            offset += len(x)
        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)
        self.valid = np.concatenate(valid)
        self.device = device
        self.xt = torch.as_tensor(self.x, device=device)
        self.yt = torch.as_tensor(self.y, device=device)

    def ids(self, clips):
        parts = [np.arange(c["start"], c["end"], dtype=np.int64) for c in clips]
        ids = np.concatenate(parts)
        return ids[self.valid[ids]]


def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def metrics(y, probs, clips, ids):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    pred = probs.argmax(1)
    subjects, cameras, clip_ids, views, groups = [np.empty(len(ids), dtype=object) for _ in range(5)]
    clip_rows, source_bins = [], defaultdict(list)
    for c in clips:
        mask = (ids >= c["start"]) & (ids < c["end"])
        if not mask.any():
            continue
        subjects[mask], cameras[mask], clip_ids[mask], views[mask], groups[mask] = c["person"], c["camera"], c["clip_id"], c["camera_view"], c["source_group_id"]
        clip_rows.append(dict(clip_id=c["clip_id"], label=c["label"], pred=int(probs[mask].mean(0).argmax())))
        source_bins[c["source_group_id"]].append((probs[mask].sum(0), int(mask.sum()), c["label"]))
    def f1(a, b):
        return float(f1_score(a, b, labels=LABELS, average="macro", zero_division=0))
    per_subject = [dict(person=p, frames=int((subjects == p).sum()), macro_f1=f1(y[subjects == p], pred[subjects == p]), accuracy=float((y[subjects == p] == pred[subjects == p]).mean())) for p in sorted(set(subjects))]
    per_camera = [dict(camera=p, view=str(views[cameras == p][0]), frames=int((cameras == p).sum()), macro_f1=f1(y[cameras == p], pred[cameras == p]), accuracy=float((y[cameras == p] == pred[cameras == p]).mean())) for p in sorted(set(cameras))]
    confidence = probs.max(1)
    ece = 0.
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= a) & (confidence < b if b < 1 else confidence <= b)
        if mask.any():
            ece += mask.mean() * abs((y[mask] == pred[mask]).mean() - confidence[mask].mean())
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    source_y = [pieces[0][2] for pieces in source_bins.values()]
    source_pred = [int((sum(p for p, n, _ in pieces) / sum(n for p, n, _ in pieces)).argmax()) for pieces in source_bins.values()]
    result = dict(frame_accuracy=float(accuracy_score(y, pred)), frame_macro_f1=f1(y, pred),
                  clip_macro_f1=f1([r["label"] for r in clip_rows], [r["pred"] for r in clip_rows]),
                  source_macro_f1=f1(source_y, source_pred),
                  subject_macro_f1=float(np.mean([s["macro_f1"] for s in per_subject])),
                  min_subject_macro_f1=float(min(s["macro_f1"] for s in per_subject)),
                  validation_loss=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()),
                  calibration_ece=float(ece), brier_score=float(np.square(probs - np.eye(6)[y]).sum(1).mean()),
                  frames=len(y), clips=len(clip_rows), source_takes=len(source_bins),
                  confusion_matrix=confusion_matrix(y, pred, labels=LABELS).tolist(),
                  per_subject=per_subject, per_camera=per_camera, classification_report=report)
    return result, pd.DataFrame(dict(label=y, pred=pred, person=subjects, camera=cameras, clip_id=clip_ids, source_group_id=groups, confidence=confidence))


def evaluate(model, data, clips):
    import torch
    ids = data.ids(clips)
    model.eval()
    with torch.inference_mode():
        transform = getattr(data, 'model_features', lambda x: x)
        probs = np.concatenate([model(transform(data.xt[torch.as_tensor(chunk, device=data.device)])).softmax(1).cpu().numpy()
                                for chunk in np.array_split(ids, max(1, math.ceil(len(ids)/4096)))])
    return metrics(data.y[ids], probs, clips, ids)


def save_metrics(result, frames, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "metrics.json", result)
    pd.DataFrame(result["classification_report"]).transpose().to_csv(folder / "classification_report.csv")
    pd.DataFrame([dict(label=k, **result["classification_report"][str(k)]) for k in LABELS]).to_csv(folder / "per_class_metrics.csv", index=False)
    pd.DataFrame(result["per_subject"]).to_csv(folder / "per_subject_metrics.csv", index=False)
    pd.DataFrame(result["per_camera"]).to_csv(folder / "camera_comparison.csv", index=False)
    frames.to_csv(folder / "frame_predictions.csv.gz", index=False, compression="gzip")
    cm = np.array(result["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set(xlabel="Predicted label", ylabel="True label", xticks=LABELS, yticks=LABELS)
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix.png", dpi=150); plt.close(fig)


def learning_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(df.epoch, df.train_loss, label="Train objective")
    if "val_loss" in df:
        axes[0].plot(df.epoch, df.val_loss, label="Validation plain CE")
        axes[1].plot(df.epoch, df.val_subject_f1, label="Validation subject Macro-F1")
        axes[1].legend()
    axes[0].legend(); axes[0].set(xlabel="Epoch", ylabel="Loss"); axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0,1))
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def class_weights(data, clips, cfg):
    import torch
    if cfg["class_weight"] == "none":
        return None
    # One effective count per ORIGINAL take/camera, unaffected by recording length.
    groups = {(c["source_group_id"], c["camera"]): c["label"] for c in clips
              if data.valid[c["start"]:c["end"]].any()}
    counts = np.bincount(list(groups.values()), minlength=6).astype(float)
    if (counts == 0).any():
        raise ValueError("Training split lacks a class")
    weights = np.sqrt(counts.mean()/counts)
    weights /= weights.mean()
    weights = np.clip(weights, .5, 2.)
    return torch.tensor(weights, dtype=torch.float32, device=data.device)


def train_epoch(model, opt, loss_fn, data, clips, cfg, epoch):
    import torch
    if cfg["quota"]:
        ids = group_sample_epoch(clips, data.valid, cfg["quota"], cfg["seed"], epoch, cfg["balance"])
    else:
        if cfg["balance"] != "none":
            raise ValueError("Balancing requires source quota")
        ids = data.ids(clips)
        np.random.default_rng(np.random.SeedSequence([cfg["seed"], epoch])).shuffle(ids)
    model.train()
    generator = torch.Generator(device=data.device).manual_seed(cfg["seed"]*1000003+epoch*1009+17)
    total = 0.
    for start in range(0, len(ids), cfg["batch_size"]):
        idx = torch.as_tensor(ids[start:start+cfg["batch_size"]], device=data.device)
        transform = getattr(data, 'training_features', augment_pose13_torch)
        x = transform(data.xt[idx], cfg, generator)
        loss = loss_fn(model(x), data.yt[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        opt.zero_grad(set_to_none=True); loss.backward()
        if cfg.get('gradient_clip',0)>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
        opt.step()
        total += float(loss.detach()) * len(idx)
    return total/max(1, len(ids)), len(ids)


def atomic_torch_save(value, path):
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def run_candidate(run, cfg, device, name, resume=False, final_epochs=None, data=None, fixed_epochs=None, experiment_dir=None):
    import torch
    from shared.mlp_model import build_classifier_model
    folder = Path(experiment_dir) if experiment_dir is not None else Path(run) / "experiments" / name
    if (folder / "result.json").exists():
        if read_json(folder / "config.json") != cfg:
            raise ValueError("Existing experiment config differs; choose new name")
        return read_json(folder / "result.json")
    if folder.exists() and not resume:
        raise FileExistsError("Incomplete run exists; use --resume: " + str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "config.json").exists() and read_json(folder / "config.json") != cfg:
        raise ValueError("Resume config differs")
    write_json(folder / "config.json", cfg)
    # Test arrays cannot be loaded by a candidate, including final training.
    data = TrainingData(run, ["train", "val"], device) if data is None else data
    train = [c for c in data.clips if c["split"] == "train" or final_epochs is not None]
    val = [c for c in data.clips if c["split"] == "val"]
    set_seed(cfg["seed"])
    model = build_classifier_model(cfg["model"], cfg["dropout"], getattr(data, 'input_dim', 52)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weights = class_weights(data, train, cfg)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg["label_smoothing"])
    write_json(folder / "training_counts.json", dict(class_weights=None if weights is None else weights.cpu().tolist(),
               train_people=sorted({c["person"] for c in train}), frames_by_label=np.bincount(data.y[data.ids(train)], minlength=6).tolist()))
    history, best_key, best_epoch, stale, start_epoch = [], None, 0, 0, 1
    last = folder / "last.pt"
    if resume and last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)  # trusted local optimizer/RNG state only
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        history, best_key, best_epoch, stale = saved["history"], saved["best_key"], saved["best_epoch"], saved["stale"]
        start_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in saved["rng_cuda"]])
    maximum = int(final_epochs or fixed_epochs or cfg["epochs"])
    for epoch in range(start_epoch, maximum + 1):
        train_loss, nsamples = train_epoch(model, opt, loss_fn, data, train, cfg, epoch)
        row = dict(epoch=epoch, train_loss=train_loss, train_samples=nsamples)
        if final_epochs is None:
            score, _ = evaluate(model, data, val)
            primary=score['frame_macro_f1'] if cfg.get('selection_metric')=='frame_macro_f1' else score['subject_macro_f1']
            key = (primary, score["min_subject_macro_f1"], score["source_macro_f1"], -score["validation_loss"])
            row.update(val_loss=score["validation_loss"], val_subject_f1=score["subject_macro_f1"], val_frame_f1=score["frame_macro_f1"], val_ece=score["calibration_ece"])
            if (fixed_epochs is None and (best_key is None or key > tuple(best_key))) or (fixed_epochs is not None and epoch == maximum):
                best_key, best_epoch, stale = key, epoch, 0
                atomic_torch_save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, model_name=cfg["model"], model_config=cfg, epoch=epoch), folder / "best.pt")
            else:
                stale += 1
        history.append(row)
        atomic_torch_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                          history=history, best_key=best_key, best_epoch=best_epoch, stale=stale,
                          rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), last)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        if epoch == 1 or epoch % 10 == 0:
            LOG.info("%s epoch %d/%d loss=%.4f val_subject_f1=%s", name, epoch, maximum, train_loss, row.get("val_subject_f1", "final_train"))
        if final_epochs is None and fixed_epochs is None and stale >= cfg["patience"]:
            break
    learning_curves(history, folder / "learning_curves.png")
    if final_epochs is not None:
        atomic_torch_save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, model_name=cfg["model"], model_config=cfg, epoch=maximum), folder / "trained.pt")
        result = dict(name=name, config=cfg, best_epoch=maximum, epochs_completed=len(history))
    else:
        saved = torch.load(folder / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        score, frames = evaluate(model, data, val)
        save_metrics(score, frames, folder)
        result = dict(name=name, config=cfg, best_epoch=best_epoch, epochs_completed=len(history), metrics=score)
    write_json(folder / "result.json", result)
    LOG.info("Completed %s: best epoch %d", name, result["best_epoch"])
    return result


def comparison(run):
    results = [read_json(p) for p in (Path(run) / "experiments").glob("*/result.json")]
    rows = []
    for r in results:
        if "metrics" not in r or r["name"].startswith("smoke"):
            continue
        rows.append(dict(experiment=r["name"], model=r["config"]["model"], seed=r["config"]["seed"],
                         best_epoch=r["best_epoch"], epochs=r["epochs_completed"],
                         **{k:v for k,v in r["metrics"].items() if isinstance(v, (float, int))}))
    df = pd.DataFrame(rows).sort_values(["subject_macro_f1", "source_macro_f1", "validation_loss", "experiment"], ascending=[False, False, True, True])
    df.to_csv(Path(run) / "comparison.csv", index=False)
    return df


def baselines(run, device, defaults=None, resume=False):
    defaults = {**DEFAULT, **(defaults or {})}
    smoke = {**defaults, "model":"mlp_small", "epochs":8, "patience":8}
    run_candidate(run, smoke, device, "smoke_small", resume)
    history = pd.read_csv(Path(run) / "experiments/smoke_small/history.csv")
    if history.train_loss.iloc[-1] >= history.train_loss.iloc[0]:
        raise RuntimeError("Smoke loss did not decrease")
    for arch in ["mlp_small", "mlp_base", "mlp_wide"]:
        cfg = {**defaults, "model":arch, "class_weight":"none", "label_smoothing":0., "quota":0, "balance":"none"}
        for key in AUGMENTATION:
            cfg.pop(key, None)
        run_candidate(run, cfg, device, "baseline_" + arch, resume)
        comparison(run)


def improvements(run, device, resume=False):
    df = comparison(run)
    best = df[df.experiment.str.startswith("baseline_")].iloc[0].experiment
    base = read_json(Path(run) / "experiments" / best / "config.json")
    # Controlled one-factor changes against the same architecture/seed baseline.
    variants = {
        "weight_sqrt":dict(class_weight="sqrt_inverse"),
        "smooth_005":dict(label_smoothing=.05),
        "smooth_010":dict(label_smoothing=.10),
        "augment":AUGMENTATION,
        "source_quota":dict(quota=48),
        "source_mild_balance":dict(quota=48, balance="sqrt_inverse"),
    }
    baseline_score = read_json(Path(run) / "experiments" / best / "result.json")["metrics"]["subject_macro_f1"]
    successes = []
    for name, change in variants.items():
        result = run_candidate(run, {**base, **change}, device, name, resume)
        if result["metrics"]["subject_macro_f1"] >= baseline_score + .005:
            successes.append((result["metrics"]["subject_macro_f1"], name, change))
        comparison(run)
    # Avoid incompatible alternatives: choose strongest smoothing/sampling setting.
    combined, seen = {}, set()
    for _, name, change in sorted(successes, reverse=True):
        family = "sampling" if name.startswith("source_") else "smoothing" if name.startswith("smooth_") else name
        if family not in seen:
            combined.update(change); seen.add(family)
    if len(seen) >= 2:
        run_candidate(run, {**base, **combined}, device, "combined", resume)
    df = comparison(run)
    candidates = df[~df.experiment.str.startswith("confirm_")].head(2).experiment.tolist()
    repeats = []
    for name in candidates:
        result = read_json(Path(run) / "experiments" / name / "result.json")
        repeats.append(dict(candidate=name, seed=result["config"]["seed"], f1=result["metrics"]["subject_macro_f1"], epoch=result["best_epoch"]))
        for seed in [303, 304]:
            cfg = {**result["config"], "seed":seed}
            repeat = run_candidate(run, cfg, device, f"confirm_{name}_seed{seed}", resume)
            repeats.append(dict(candidate=name, seed=seed, f1=repeat["metrics"]["subject_macro_f1"], epoch=repeat["best_epoch"]))
    confirmation = pd.DataFrame(repeats)
    confirmation.to_csv(Path(run) / "seed_confirmation.csv", index=False)
    ranking = confirmation.groupby("candidate").agg(mean_f1=("f1","mean"), std_f1=("f1","std"), median_epoch=("epoch","median")).reset_index().sort_values(["mean_f1", "candidate"], ascending=[False, True])
    ranking.to_csv(Path(run) / "confirmed_ranking.csv", index=False)
    winner = str(ranking.iloc[0].candidate)
    config = read_json(Path(run) / "experiments" / winner / "config.json")
    selection = dict(candidate=winner, config=config, final_seed=10302,
                     final_epochs=max(1, int(round(ranking.iloc[0].median_epoch))),
                     confirmation=ranking.to_dict("records"), test_accessed=False,
                     selection_metric="validation mean per-subject six-class Macro-F1; top two candidates across seeds 302,303,304",
                     meaningful_improvement_threshold=.005)
    selection_path = Path(run) / "selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise RuntimeError("Frozen selection changed; use a new training run")
    write_json(selection_path, selection)
    comparison(run)
    return selection


def finalize(run, model_root, device, resume=False):
    import torch
    from shared.mlp_model import PostureMLP
    run = Path(run)
    if (run / "final_model.json").exists():
        return Path(read_json(run / "final_model.json")["path"])
    selection = read_json(run / "selection.json")
    cfg = {**selection["config"], "seed":selection["final_seed"]}
    result = run_candidate(run, cfg, device, "final_retrain", resume, selection["final_epochs"])
    info = read_json(run / "split_manifest.json")
    train_people = sorted(info["train_people"] + info["validation_people"])
    if set(train_people) & set(info["test_people"]):
        raise RuntimeError("Final subject overlap")
    marker = run / "test_evaluation_started.json"
    if marker.exists():
        raise RuntimeError("Final test already started. Inspect saved outputs; automatic second evaluation is prohibited.")
    folder = version_dir(Path(model_root) / cfg["model"])
    trained = torch.load(run / "experiments/final_retrain/trained.pt", weights_only=True, map_location="cpu")
    contract = read_json(run / "feature_contract.json")
    label_map = read_json(run / "snapshot/config/labels.json")
    names = [r["name"] for r in sorted(label_map["labels"], key=lambda r:r["id"])]
    trained.update(stage="final", feature_contract=contract, label_names=names,
                   train_people=train_people, dataset_id=contract["dataset_id"], split_manifest=info)
    atomic_torch_save(trained, folder / "model_final.pt")
    write_json(folder / "model_config.json", cfg, exclusive=True)
    write_json(folder / "feature_contract.json", contract, exclusive=True)
    write_json(folder / "label_map.json", label_map, exclusive=True)
    write_json(folder / "split_manifest.json", info, exclusive=True)
    summary = dict(selection=selection, final_epochs=result["best_epoch"], train_people=train_people,
                   test_people=info["test_people"], checkpoint_sha256=file_sha256(folder / "model_final.pt"),
                   dataset_id=contract["dataset_id"], training_run=str(run.resolve()), test_evaluation_count=0)
    write_json(folder / "training_summary.json", summary, exclusive=True)
    # Persist the final checkpoint before one and only one test prediction pass.
    write_json(marker, dict(model=str(folder.resolve()), checkpoint_sha256=summary["checkpoint_sha256"], started_utc=datetime.now(timezone.utc).isoformat()), exclusive=True)
    data = TrainingData(run, ["test"], device)
    model = PostureMLP(cfg["model"], cfg["dropout"]).to(device)
    model.load_state_dict(trained["model_state_dict"], strict=True)
    score, frames = evaluate(model, data, data.clips)
    score["test_loss"] = score.pop("validation_loss")
    save_metrics(score, frames, folder / "test")
    write_json(folder / "test_metrics.json", score, exclusive=True)
    summary["test_evaluation_count"] = 1
    summary["test_metrics"] = score
    write_json(folder / "training_summary.json", summary)
    shutil.copyfile(run / "experiments/final_retrain/history.csv", folder / "history.csv")
    shutil.copyfile(run / "experiments/final_retrain/learning_curves.png", folder / "learning_curves.png")
    write_json(run / "training_summary.json", summary)
    write_json(run / "final_model.json", dict(path=str(folder.resolve())), exclusive=True)
    return folder


def export_and_validate(run):
    import torch
    from scripts.export_onnx import export
    from inference.pipeline import PostureClassifier
    run = Path(run)
    model_dir = Path(read_json(run / "final_model.json")["path"])
    if not (model_dir / "model.onnx").exists():
        export(model_dir)
    # Real train/validation observations only; no second test prediction pass.
    data = TrainingData(run, ["train", "val"], "cpu")
    pts = data.x[data.ids(data.clips)]
    from shared.mlp_model import PostureMLP
    ckpt = torch.load(model_dir / "model_final.pt", map_location="cpu", weights_only=True)
    model = PostureMLP(ckpt["model_name"], ckpt["model_config"]["dropout"]).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    classifier = PostureClassifier(model_dir, "onnx", "cpu")
    maximum, logit_max = 0., 0.
    with torch.inference_mode():
        for start in range(0, len(pts), 1024):
            features = pts[start:start+1024]
            a = model(torch.from_numpy(features))
            b = classifier.session.run(["logits"], {"features":features})[0]
            p = torch.from_numpy(b).softmax(1).numpy()
            maximum = max(maximum, float(np.abs(a.softmax(1).numpy()-p).max()))
            logit_max = max(logit_max, float(np.abs(a.numpy()-b).max()))
    if maximum > 1e-5 or not np.isfinite(maximum):
        raise RuntimeError(f"Real feature ONNX probability mismatch: {maximum}")
    # Both deployment loader paths, six probabilities, matching contract.
    pt = PostureClassifier(model_dir, "pt", "cpu")
    for x in pts[:16]:
        p = pt.predict(x)
        np.testing.assert_allclose(p, classifier.predict(x), atol=1e-5, rtol=1e-5)
        if p.shape != (6,) or not np.isclose(p.sum(), 1):
            raise RuntimeError("Invalid deployment probabilities")
    report = dict(verified=True, real_features=len(pts), max_abs_probability_difference=maximum,
                  max_abs_logit_difference=logit_max, tolerance=1e-5,
                  feature_contract=FEATURE_VERSION, checkpoint_sha256=file_sha256(model_dir / "model_final.pt"))
    write_json(model_dir / "onnx_real_parity.json", report)
    return report
