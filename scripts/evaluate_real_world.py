#!/usr/bin/env python3
"""Guarded evaluation of newly consented, isolated labeled external recordings.

This command never accesses a live camera. Live functional inference remains in
inference/run.py and requires the user's later explicit authorization.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd

from shared import real_training as rt
from shared.dataset_pipeline import file_sha256


def validate_external(root,manifest,training_root,inventory,known_people):
    root=Path(root).resolve();training_root=Path(training_root).resolve()
    if root==training_root or root.is_relative_to(training_root):raise ValueError('External dataset must be physically isolated from training')
    if manifest.get('dataset_role')!='independent_external_test' or manifest.get('consent_confirmed') is not True:
        raise ValueError('Independent dataset role and explicit volunteer consent confirmation required')
    recordings=manifest.get('recordings',[])
    if not recordings:raise ValueError('No new external recordings')
    training_hashes={r['sha256'] for name,r in inventory.items() if name.endswith('.mp4')}
    ids=set();video_hashes=set();rows=[]
    for row in recordings:
        required=['person_id','session_id','take_id','camera_id','label','video_path','timestamps_path']
        if any(k not in row for k in required):raise ValueError('External recording lineage missing')
        if row['person_id'] in set(known_people):raise ValueError('External subjects must be new volunteers')
        if int(row['label']) not in range(6):raise ValueError('External label must be 0-5')
        identity=(row['person_id'],row['session_id'],row['take_id'],row['camera_id'])
        if identity in ids:raise ValueError('Duplicate external take/camera identity')
        ids.add(identity)
        video=rt.inside(root,row['video_path']);timestamps=rt.inside(root,row['timestamps_path'])
        if not video.is_file() or not timestamps.is_file():raise FileNotFoundError('Missing external video or capture timestamp CSV')
        if video.resolve().is_relative_to(training_root):raise ValueError('Training video cannot be used as external test')
        sha=file_sha256(video)
        if sha in training_hashes or sha in video_hashes:raise ValueError('Copied training or duplicate video is not an independent recording')
        video_hashes.add(sha)
        rows.append(dict(**row,video_sha256=sha,timestamp_sha256=file_sha256(timestamps)))
    return rows


def evaluate(args):
    from inference.pipeline import PostureClassifier
    from inference.run import run_file
    manifest=rt.read_json(args.manifest)
    config=rt.read_json(args.training_run/'run_config.json')
    catalog=pd.read_csv(args.training_run/'split_manifest.csv')
    recordings=validate_external(args.data,manifest,config['raw_root'],
                 rt.read_json(args.training_run/'audit/source_inventory.json'),catalog.person_id.unique())
    model_sha=file_sha256(args.model/'model_final.pt')
    # One declared frozen external dataset/model combination; this marker is
    # written before predictions and prevents silently rerunning a failed test.
    fingerprint=file_sha256(args.manifest)+'_'+model_sha
    registry=args.output/'evaluation_registry';registry.mkdir(parents=True,exist_ok=True)
    rt.write_json(registry/(fingerprint+'.json'),dict(model_sha256=model_sha,manifest_sha256=file_sha256(args.manifest),
                  policy='Untouched external evaluation; further tuning makes this dataset development evidence.'),exclusive=True)
    out=rt.version_dir(args.output,'external_v')
    classifier=PostureClassifier(args.model,args.backend,'cpu')
    all_frames=[];clip_rows=[];offset=0
    options=Namespace(pose=ROOT/'artifacts/pretrained/yolo11n-pose.pt',device=args.device,stable_frames=5,min_conf=.55,
                      display=args.display,max_frames=0,timestamps=None)
    for index,row in enumerate(recordings):
        folder=out/f'clip_{index:03d}';folder.mkdir();options.timestamps=rt.inside(args.data,row['timestamps_path'])
        run_file(rt.inside(args.data,row['video_path']),'video',options,classifier,folder)
        frames=pd.read_csv(folder/'frames.csv');performance=rt.read_json(folder/'performance.json')
        frames['person']=row['person_id'];frames['camera']=row['camera_id'];frames['label']=int(row['label'])
        frames['clip_id']=str(index);frames['source_group_id']=f'{row["person_id"]}/{row["session_id"]}/{row["label"]}/{row["take_id"]}'
        frames['pred']=frames.class_id
        all_frames.append(frames)
        clip_rows.append(dict(clip_id=str(index),person=row['person_id'],camera=row['camera_id'],label=row['label'],**performance))
    frames=pd.concat(all_frames,ignore_index=True)
    available=frames[frames.pred.notna() & (frames.keypoints_valid>=4)].copy();available['pred']=available.pred.astype(int)
    if available.empty:raise RuntimeError('No valid external predictions; coverage zero, no accuracy score can be claimed')
    probability=available[[f'p{i}' for i in range(6)]].to_numpy(np.float32)
    clips=[];cursor=0
    for clip,piece in available.groupby('clip_id',sort=False):
        first=piece.iloc[0]
        clips.append(dict(start=cursor,end=cursor+len(piece),person=first.person,camera=first.camera,camera_view=first.camera,
                         label=int(first.label),clip_id=clip,source_group_id=first.source_group_id));cursor+=len(piece)
    result,predictions=rt.metrics(available.label.to_numpy(np.int64),probability,clips,np.arange(len(available)))
    result.update(evaluation_scope='GENUINE LABELED INDEPENDENT EXTERNAL TEST',
                  total_video_frames=len(frames),prediction_coverage=len(available)/len(frames),model_sha256=model_sha,
                  quality_policy='All frames retained in logs; quantitative target requires >=4 measured joints and an available prediction.',
                  raw_prediction_frames=int(frames.pred.notna().sum()),
                  frozen_manifest_sha256=file_sha256(args.manifest),independent_subjects=sorted(set(available.person)))
    # Fixed-six subject F1 remains recorded; also show F1 over each subject's
    # actually present ground-truth classes to avoid absent-label confusion.
    from sklearn.metrics import f1_score
    result['per_subject_present_label_f1']={p:float(f1_score(g.label,g.pred,labels=sorted(g.label.unique()),average='macro',zero_division=0)) for p,g in available.groupby('person')}
    rt.save_metrics(result,predictions,out)
    frames.to_csv(out/'all_frames_with_abstentions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(clip_rows).to_csv(out/'clip_performance.csv',index=False)
    rt.write_json(out/'external_recording_provenance.json',recordings,exclusive=True)
    print(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True,help='Isolated root containing NEW consented volunteer videos')
    ap.add_argument('--manifest',type=Path,required=True,help='JSON: dataset_role, consent_confirmed, recordings with person/session/take/camera/label/video_path/timestamps_path')
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--training-run',type=Path,default=ROOT/'artifacts/experiments/take_level_gru/v001')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/external_evaluations')
    ap.add_argument('--backend',choices=['pt','onnx'],default='onnx')
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--display',action='store_true')
    args=ap.parse_args();evaluate(args)


if __name__=='__main__':main()
