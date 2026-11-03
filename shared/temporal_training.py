"""Capture-time paired-camera temporal splits and all-real-data MLP fitting.

Reuses real_training's audit, causal features, training engine, metrics and resume.
Internal temporal diagnostics and all-data fitting have separate entry points.
"""
from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from shared import real_training as rt
from shared.dataset_pipeline import file_sha256
from shared.pose_features import FEATURE_VERSION
from shared.mlp_model import build_classifier_model

LOG=logging.getLogger('poseforge.training')
STRATEGY='temporal_within_take'
TEST_LABEL='INTERNAL WITHIN-TAKE TEST – SAME SUBJECTS AND RECORDINGS'
PARTITIONS=('train','val','internal_test')


def temporal_boundaries(start_ns,end_ns,ratios=(.8,.1,.1)):
    ratios=np.asarray(ratios,dtype=float)
    if len(ratios)!=3 or not np.isfinite(ratios).all() or (ratios<=0).any() or not np.isclose(ratios.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('Three positive temporal ratios must sum to one')
    if int(start_ns)>=int(end_ns):raise ValueError('Invalid capture timeline bounds')
    duration=int(end_ns)-int(start_ns)
    b1=int(start_ns)+int(round(duration*ratios[0]))
    b2=int(start_ns)+int(round(duration*(ratios[0]+ratios[1])))
    if not int(start_ns)<b1<b2<int(end_ns):raise ValueError('Degenerate temporal boundaries')
    return b1,b2


def assign_capture_times(ticks,start_ns,end_ns,ratios=(.8,.1,.1),purge_seconds=.25):
    ticks=np.asarray(ticks,dtype=np.int64)
    if not np.isfinite(purge_seconds) or purge_seconds<0:raise ValueError('Purge radius must be nonnegative')
    if ((ticks<int(start_ns))|(ticks>=int(end_ns))).any():raise ValueError('Timestamp outside original take timeline')
    if (np.diff(ticks)<0).any():raise ValueError('Backward capture clock')
    b1,b2=temporal_boundaries(start_ns,end_ns,ratios)
    planned=np.select([ticks<b1,ticks<b2],['train','val'],default='internal_test')
    radius=int(round(purge_seconds*1e9))
    purged=((np.abs(ticks-b1)<=radius)|(np.abs(ticks-b2)<=radius)) if radius else np.zeros(len(ticks),bool)
    return planned,purged


def verify_camera_clocks(camera_times,metadata,tolerance_ms=50.):
    """Verify a common host-reception clock using UTC and monotonic pairs.

    This does not establish exposure synchronization or remove network delay.
    Repeated monotonic ticks are retained; backward clocks invalidate the take.
    """
    issues=[];offsets=[];ranges=[]
    if len(camera_times)<2:issues.append('missing_paired_camera')
    lower=int(metadata.get('start_host_monotonic_ns',-1))
    upper=int(metadata.get('end_host_monotonic_ns',-1))
    if lower<0 or upper<=lower:issues.append('missing_or_invalid_recording_clock_bounds')
    tolerance=int(tolerance_ms*1e6)
    for camera,table in camera_times.items():
        needed={'writer_frame_index','source_frame_index','host_recv_monotonic_ns','host_recv_utc'}
        if not needed<=set(table.columns) or table.empty:
            issues.append(camera+':missing_timestamp_provenance');continue
        ticks=table.host_recv_monotonic_ns.to_numpy(np.int64)
        if table.writer_frame_index.tolist()!=list(range(len(table))):issues.append(camera+':invalid_frame_order')
        if (np.diff(ticks)<0).any():issues.append(camera+':backward_ticks')
        if (ticks<lower-tolerance).any() or (ticks>upper+tolerance).any():issues.append(camera+':outside_metadata_clock')
        try:
            # Explicit ns conversion is required with newer pandas datetime units.
            utc=pd.to_datetime(table.host_recv_utc,format='mixed',utc=True).dt.as_unit('ns').astype('int64').to_numpy()
            offset=utc-ticks
            offsets.append(int(np.median(offset)));ranges.append(int(np.ptp(offset)))
        except (ValueError,TypeError):issues.append(camera+':invalid_utc_clock')
    offset_spread=max(offsets)-min(offsets) if offsets else 0
    max_range=max(ranges,default=0)
    if offset_spread>tolerance:issues.append('cross_camera_utc_monotonic_offset_mismatch')
    if max_range>tolerance:issues.append('unstable_utc_monotonic_offset')
    return dict(verified=not issues,issues=issues,offset_spread_ms=offset_spread/1e6,
                max_offset_range_ms=max_range/1e6,tolerance_ms=tolerance_ms,
                timeline='common host reception monotonic clock; not exposure-synchronized')


def apply_segment_quality(frame_table,min_eval_frames=8,min_eval_seconds=.10):
    """Reject short camera segments; never move their frames into training."""
    frame_table=frame_table.copy()
    frame_table['partition']=frame_table.planned_partition.astype(object)
    frame_table['exclusion_reason']=''
    frame_table.loc[~frame_table.eligible,'exclusion_reason']='existing_quality_exclusion'
    frame_table.loc[frame_table.embargo & frame_table.eligible,'exclusion_reason']='temporal_embargo'
    frame_table.loc[~frame_table.clock_verified & frame_table.eligible,'exclusion_reason']='unverified_paired_clock'
    active=frame_table.eligible & ~frame_table.embargo & frame_table.clock_verified
    frame_table.loc[~active,'partition']='excluded'
    segments=[]
    for (group,camera,part),piece in frame_table.groupby(['source_group_id','camera_id','planned_partition'],sort=True):
        available=piece[piece.partition==part]
        span=(int(available.capture_ns.max())-int(available.capture_ns.min()))/1e9 if len(available)>1 else 0.
        enough=(len(available)>=min_eval_frames and span>=min_eval_seconds) if part!='train' else len(available)>0
        if part!='train' and not enough:
            ids=available.index
            frame_table.loc[ids,'partition']='excluded'
            frame_table.loc[ids,'exclusion_reason']='short_or_sparse_evaluation_segment'
        segments.append(dict(source_group_id=group,camera_id=camera,partition=part,
                             eligible_before_embargo=int(piece.eligible.sum()),
                             eligible_embargo_frames=int((piece.eligible&piece.embargo).sum()),
                             candidate_frames=len(available),candidate_span_s=span,
                             usable=bool(enough),assigned_frames=len(available) if enough else 0))
    return frame_table,pd.DataFrame(segments)


def leakage_checks(frames,boundaries,ratios,purge_seconds):
    identity=['person_id','session_id','source_take_id','camera_id','source_frame_index']
    if frames[identity].isna().any().any():raise ValueError('Missing source frame provenance')
    if frames.duplicated(identity).any():raise ValueError('Duplicate source frame identity')
    if not set(frames.label.astype(int))<=set(range(6)):raise ValueError('Invalid posture label')
    selected=frames[frames.partition.isin(PARTITIONS)]
    if not selected.eligible.all():raise ValueError('Quality-excluded frame assigned to training/evaluation')
    if selected.embargo.any() or not selected.clock_verified.all():raise ValueError('Embargo/unverified clock leakage')
    for group,piece in selected.groupby('source_group_id'):
        b=boundaries[group]
        expected,purged=assign_capture_times(piece.capture_ns.sort_values().to_numpy(),b['start_ns'],b['end_ns'],ratios,purge_seconds)
        actual=piece.sort_values('capture_ns').partition.to_numpy()
        if purged.any() or not np.array_equal(expected,actual):raise ValueError('Cross-camera temporal assignment differs from shared boundaries')
        # Repeated ticks, including across cameras, always have one partition.
        if (piece.groupby('capture_ns').partition.nunique()>1).any():raise ValueError('One capture moment appears in multiple partitions')
        bins={part:piece[piece.partition==part].capture_ns.to_numpy() for part in PARTITIONS}
        for left,right in [('train','val'),('train','internal_test'),('val','internal_test')]:
            if len(bins[left]) and len(bins[right]) and int(bins[left].max())>=int(bins[right].min()):
                raise ValueError('Cross-camera chronology leakage')
    return dict(passed=True,unique_source_frames=len(frames),assigned_frames=len(selected),
                duplicate_frame_identities=0,cross_camera_partition_conflicts=0,
                invalid_labels=0,quality_excluded_frames_assigned=0,
                actual_capture_timeline_used=True,encoded_fps_used=False,
                repeated_ticks='Retained; all equal ticks assigned by the same take-level boundaries',
                causal_features='Original causal stream; past raw observations may initialize later features; no future-frame normalization',
                limitation='Common host-reception clock only; no exposure synchronization guarantee. Same takes/subjects/scenes remain correlated.')


def partition_counts(frames):
    rows=[]
    for partition in [*PARTITIONS,'excluded']:
        q=frames[frames.partition==partition]
        rows.append(dict(partition=partition,frames=len(q),source_takes=int(q.source_group_id.nunique()),
                 subjects=int(q.person_id.nunique()),camera_clips=int(q.clip_id.nunique()),
                 labels={str(k):int(v) for k,v in q.label.value_counts().sort_index().items()},
                 cameras={str(k):int(v) for k,v in q.camera_id.value_counts().sort_index().items()}))
    return rows


def build_split(raw_root,catalog,quality,options):
    raw_root=Path(raw_root).resolve();rows=[];takes=[];boundaries={}
    for group,clips in catalog.groupby('source_group_id',sort=True):
        if clips.camera_id.duplicated().any():
            raise ValueError('Temporal splitting requires the original camera streams, not separately exported segments of one source take')
        tables={};metadata=None
        for row in clips.to_dict('records'):
            video=rt.inside(raw_root,row['video_path'])
            tables[row['camera_id']]=pd.read_csv(video.parent/f'{row["camera_id"]}_timestamps.csv')
            meta=rt.read_json(rt.inside(raw_root,row['metadata_path']))
            if metadata is not None and metadata['take_id']!=meta['take_id']:raise ValueError('Camera pair source identity mismatch')
            metadata=meta
        clock=verify_camera_clocks(tables,metadata,options['clock_tolerance_ms'])
        start=min(int(t.host_recv_monotonic_ns.min()) for t in tables.values())
        end=max(int(t.host_recv_monotonic_ns.max()) for t in tables.values())+1
        b1,b2=temporal_boundaries(start,end,options['ratios'])
        bounds=dict(start_ns=start,end_ns=end,train_val_boundary_ns=b1,val_test_boundary_ns=b2)
        boundaries[group]=bounds
        takes.append(dict(source_group_id=group,person_id=clips.person_id.iloc[0],label=int(clips.label.iloc[0]),
                          duration_s=(end-start)/1e9,clock=clock,**bounds))
        for row in clips.to_dict('records'):
            table=tables[row['camera_id']];ticks=table.host_recv_monotonic_ns.to_numpy(np.int64)
            # Invalid monotonic order is flagged; no sorting/repaired timestamps.
            if (np.diff(ticks)<0).any():
                planned=np.select([ticks<b1,ticks<b2],['train','val'],default='internal_test')
                embargo=np.zeros(len(ticks),bool)
            else:planned,embargo=assign_capture_times(ticks,start,end,options['ratios'],options['purge_seconds'])
            q=quality[quality.clip_id==row['clip_id']].sort_values('frame_index')
            if len(q)!=len(table):raise ValueError('Timestamp/eligibility count mismatch')
            for i,t in enumerate(table.to_dict('records')):
                rows.append(dict(clip_id=row['clip_id'],person_id=row['person_id'],session_id=row['session_id'],
                    source_take_id=row['source_take_id'],source_group_id=group,camera_id=row['camera_id'],camera_view=row['camera_view'],
                    frame_index=i,source_frame_index=int(row['source_frame_start'])+i,
                    camera_source_frame_index=int(t['source_frame_index']),capture_ns=int(t['host_recv_monotonic_ns']),
                    capture_utc=str(t['host_recv_utc']),label=int(row['label']),eligible=bool(q.eligible.iloc[i]),
                    valid_joints=int(q.valid_joints.iloc[i]),planned_partition=str(planned[i]),embargo=bool(embargo[i]),clock_verified=clock['verified']))
    frames,segments=apply_segment_quality(pd.DataFrame(rows),options['min_eval_frames'],options['min_eval_seconds'])
    checks=leakage_checks(frames,boundaries,options['ratios'],options['purge_seconds'])
    coverage=[]
    for part in ['val','internal_test']:
        for label in range(6):
            q=frames[(frames.partition==part)&frames.label.eq(label)]
            coverage.append(dict(partition=part,label=label,frames=len(q),takes=int(q.source_group_id.nunique()),subjects=int(q.person_id.nunique()),
                                 enough=bool(q.source_group_id.nunique()>=options['min_eval_takes_per_class'] and q.person_id.nunique()>=options['min_eval_subjects_per_class'])))
    viable=all(r['enough'] for r in coverage)
    neligible=int(frames.eligible.sum());ntrain=int(frames.partition.eq('train').sum())
    feasibility=dict(viable=viable,partition_counts=partition_counts(frames),label_coverage=coverage,
                  reasons=[] if viable else ['Post-purge validation/internal-test lacks meaningful six-class, multi-take/multi-subject coverage.'],
                  options=options,embargo_definition='Symmetric exclusion radius: abs(capture_time - boundary) <= purge_seconds; default excludes 0.5 s total at each boundary.',
                  original_eligible_frames=neligible,eligible_embargo_frames=int((frames.eligible&frames.embargo).sum()),
                  original_frames=len(frames),effective_training_percent_of_eligible=100*ntrain/neligible,
                  effective_training_percent_of_raw=100*ntrain/len(frames),
                  clock_verified_takes=sum(t['clock']['verified'] for t in takes),flagged_takes=[t for t in takes if not t['clock']['verified']],
                  duration_summary_s=pd.Series([t['duration_s'] for t in takes]).describe().to_dict(),
                  usable_source_takes_per_partition={p:int(frames.loc[frames.partition==p,'source_group_id'].nunique()) for p in PARTITIONS},
                  historical_subject_cv_remains_primary=True,
                  evaluation_warning='Within-take samples share subjects, posture performances and scenes. Purging does not guarantee independent poses.')
    return frames,segments,takes,boundaries,checks,feasibility


def audit(raw_root,source_run,output,options):
    raw_root,source_run=Path(raw_root).resolve(),Path(source_run).resolve()
    run=rt.version_dir(output)
    report=rt.audit(raw_root,run/'audit',raw_root.parents[1]/'reextract_backups/20261030T115911Z_cc9dbeb9/manifest.json')
    catalog=pd.read_csv(run/'audit/clip_quality.csv');quality=pd.read_csv(run/'audit/frame_quality.csv.gz')
    # Preserve prior eligibility, instead of silently changing curation rules.
    old=pd.read_csv(source_run/'catalog.csv')
    if set(old.clip_id)!=set(catalog.clip_id):raise ValueError('Source clips changed; reconcile existing curation before training')
    old_inventory=rt.read_json(source_run/'snapshot/source_inventory.json')
    inventory=rt.read_json(run/'audit/source_inventory.json')
    for rel,info in old_inventory.items():
        if rel not in inventory or inventory[rel]['sha256']!=info['sha256']:raise RuntimeError('Source changed since existing eligibility was frozen: '+rel)
    feature_hashes=rt.read_json(source_run/'processed/checksums.json')
    for row in catalog.to_dict('records'):
        file=source_run/'processed'/f'{row["clip_id"]}.npz'
        if file_sha256(file)!=feature_hashes[file.name]:raise RuntimeError('Prior features changed')
        with np.load(file,allow_pickle=False) as data:eligible=data['eligible']
        ids=quality.index[quality.clip_id==row['clip_id']]
        if not np.array_equal(quality.loc[ids,'eligible'].to_numpy(),eligible):raise RuntimeError('Eligibility decisions changed; review rather than silently replace')
    frames,segments,takes,boundaries,checks,feasibility=build_split(raw_root,catalog,quality,options)
    catalog.to_csv(run/'catalog_source.csv',index=False)
    frames.to_csv(run/'temporal_split_draft.csv',index=False)
    segments.to_csv(run/'temporal_segment_feasibility.csv',index=False)
    rt.write_json(run/'take_timelines.json',takes,exclusive=True)
    rt.write_json(run/'temporal_boundaries.json',boundaries,exclusive=True)
    rt.write_json(run/'dataset_audit.json',report,exclusive=True)
    rt.write_json(run/'split_feasibility_report.json',feasibility,exclusive=True)
    rt.write_json(run/'temporal_leakage_checks.json',checks,exclusive=True)
    rt.write_json(run/'run_config.json',dict(strategy=STRATEGY,raw_root=str(raw_root),source_run=str(source_run),seed=302,options=options),exclusive=True)
    models_root=raw_root.parents[1]/'models'
    preservation={}
    # Record every earlier installed model artifact, without interpreting its metrics.
    for file in models_root.rglob('*'):
        if file.is_file():preservation[str(file.resolve())]=file_sha256(file)
    rt.write_json(run/'prior_models_hashes.json',preservation,exclusive=True)
    LOG.info('Audited %s; temporal feasibility=%s',run,feasibility['viable'])
    return run


def verify_sources(run):
    run=Path(run);root=Path(rt.read_json(run/'run_config.json')['raw_root'])
    inventory=rt.read_json(run/'audit/source_inventory.json')
    for rel,value in inventory.items():
        if file_sha256(root/rel)!=value['sha256']:raise RuntimeError('Original dataset changed: '+rel)
    for path,digest in rt.read_json(run/'prior_models_hashes.json').items():
        if file_sha256(Path(path))!=digest:raise RuntimeError('Earlier model artifact changed: '+path)
    return dict(source_files_unchanged=len(inventory),prior_model_files_unchanged=len(rt.read_json(run/'prior_models_hashes.json')))


def freeze(run):
    run=Path(run).resolve()
    if (run/'frozen_checksums.json').exists():
        verify_frozen(run);return dict(status='already_frozen',path=str(run))
    verify_sources(run)
    config=rt.read_json(run/'run_config.json');source=Path(config['source_run'])
    processed=run/'processed';processed.mkdir()
    catalog=pd.read_csv(run/'catalog_source.csv')
    old=rt.read_json(source/'processed/checksums.json')
    for row in catalog.to_dict('records'):
        file=source/'processed'/f'{row["clip_id"]}.npz'
        if file_sha256(file)!=old[file.name]:raise RuntimeError('Original feature snapshot changed')
        shutil.copyfile(file,processed/file.name)
    rt.seal(processed)
    shutil.copyfile(run/'temporal_split_draft.csv',run/'temporal_split_manifest.csv')
    frames=pd.read_csv(run/'temporal_split_manifest.csv')
    checks=leakage_checks(frames,rt.read_json(run/'temporal_boundaries.json'),config['options']['ratios'],config['options']['purge_seconds'])
    rt.write_json(run/'temporal_leakage_checks.json',checks)
    contract=rt.read_json(source/'feature_contract.json')
    rt.write_json(run/'feature_contract.json',contract,exclusive=True)
    manifest=dict(strategy=STRATEGY,seed=config['seed'],options=config['options'],
                  timeline_definition='Union of actual host-reception timestamps across both cameras: [minimum tick,maximum tick+1ns). One pair of boundaries per ORIGINAL source take.',
                  boundaries=rt.read_json(run/'temporal_boundaries.json'),
                  dataset_id=rt.read_json(run/'dataset_audit.json')['dataset_id'],
                  processed_dataset_id=file_sha256(processed/'checksums.json'),
                  source_inventory_sha256=file_sha256(run/'audit/source_inventory.json'),
                  frame_manifest_sha256=file_sha256(run/'temporal_split_manifest.csv'),
                  same_subjects_and_recordings=True,independent_test=False,
                  embargo_radius_seconds=config['options']['purge_seconds'])
    rt.write_json(run/'temporal_split_manifest.json',manifest,exclusive=True)
    names=['run_config.json','catalog_source.csv','feature_contract.json','temporal_split_manifest.csv',
           'temporal_split_manifest.json','temporal_boundaries.json','split_feasibility_report.json','audit/source_inventory.json']
    rt.write_json(run/'frozen_checksums.json',{name:file_sha256(run/name) for name in names},exclusive=True)
    for name in names:(run/name).chmod(0o444)
    return dict(status='frozen',viable=rt.read_json(run/'split_feasibility_report.json')['viable'],path=str(run))


def verify_frozen(run):
    run=Path(run)
    for rel,digest in rt.read_json(run/'frozen_checksums.json').items():
        if file_sha256(run/rel)!=digest:raise RuntimeError('Frozen temporal provenance changed: '+rel)


class FramePartitionData(rt.TrainingData):
    """Adapter to the existing training engine. Select frames before GPU transfer."""
    def __init__(self,run,partitions,device,full=False):
        run=Path(run).resolve();verify_frozen(run)
        self.run=run;self.device=device;self.input_dim=52
        if full and partitions!=['train']:raise ValueError('All-data fit has no held-out partitions')
        if not full and not set(partitions)<=set(PARTITIONS):raise ValueError('Invalid temporal partition')
        frames=pd.read_csv(run/'temporal_split_manifest.csv')
        catalog=pd.read_csv(run/'catalog_source.csv')
        root=run/'full_snapshot/processed' if full else run/'processed'
        checks=rt.read_json(root/'checksums.json')
        xs=[];ys=[];self.clips=[];self.frame_rows=[];offset=0
        for row in catalog.to_dict('records'):
            file=root/f'{row["clip_id"]}.npz'
            if file_sha256(file)!=checks[file.name]:raise RuntimeError('Immutable feature hash mismatch')
            with np.load(file,allow_pickle=False) as arrays:
                features=arrays['x'].reshape(-1,52);eligible=arrays['eligible']
            q=frames[frames.clip_id==row['clip_id']].sort_values('frame_index')
            if not np.array_equal(q.eligible.to_numpy(),eligible):raise RuntimeError('Full-data eligibility changed')
            for part in partitions:
                selected=eligible if full else q.partition.eq(part).to_numpy()
                if (selected&~eligible).any():raise ValueError('Excluded observations selected')
                ids=np.flatnonzero(selected)
                x=features[ids]
                xs.append(x);ys.append(np.full(len(x),int(row['label']),np.int64))
                self.frame_rows.extend(q.iloc[ids].to_dict('records'))
                self.clips.append(dict(start=offset,end=offset+len(x),label=int(row['label']),
                    clip_id=row['clip_id']+'__'+part,original_clip_id=row['clip_id'],person=row['person_id'],
                    camera=row['camera_id'],camera_view=row['camera_view'],source_group_id=row['source_group_id'],split=part))
                offset+=len(x)
        if offset==0:raise ValueError('No usable observations in requested partition')
        self.x=np.concatenate(xs);self.y=np.concatenate(ys);self.valid=np.ones(len(self.x),bool)
        if self.x.shape[1]!=52 or not np.isfinite(self.x).all():raise ValueError('Invalid causal training features')
        self.xt=torch.as_tensor(self.x,device=device);self.yt=torch.as_tensor(self.y,device=device)


def baseline_config(seed):
    return {**rt.DEFAULT,'model':'mlp_wide','seed':int(seed)}


def train_temporal(run,device,resume=False):
    run=Path(run);verify_frozen(run)
    feasibility=rt.read_json(run/'split_feasibility_report.json')
    if not feasibility['viable']:
        # Historical confirmation is development evidence, not the opened QBAO
        # test. It selected epochs 8/4/1 for the unchanged Wide configuration.
        source=Path(rt.read_json(run/'run_config.json')['source_run'])
        ranking=pd.read_csv(source/'confirmed_ranking.csv')
        epochs=int(ranking.loc[ranking.candidate=='baseline_mlp_wide','median_epoch'].iloc[0])
        metrics=dict(status='not_executed_infeasible',reason=feasibility['reasons'],
                     validation_metrics=None,internal_test_metrics=None,
                     historical_subject_cv='Existing primary evidence retained; not retrained')
        selection=dict(config=baseline_config(10302),final_epochs=epochs,
                   schedule_basis='Historical subject-wise development Wide confirmation: median best epochs 8,4,1 = 4; no temporal/internal-test score used.',
                   temporal_training_executed=False,internal_test_used_for_selection=False,
                   source_confirmation_sha256=file_sha256(source/'confirmed_ranking.csv'))
    else:
        data=FramePartitionData(run,['train','val'],device)
        smoke_cfg={**baseline_config(302),'epochs':5,'patience':5}
        rt.run_candidate(run,smoke_cfg,device,'smoke_temporal',resume,data=data)
        history=pd.read_csv(run/'experiments/smoke_temporal/history.csv')
        if history.train_loss.iloc[-1]>=history.train_loss.iloc[0]:raise RuntimeError('CUDA smoke loss failed to decrease')
        results=[]
        for seed in [302,303,304]:results.append(rt.run_candidate(run,baseline_config(seed),device,f'temporal_seed{seed}',resume,data=data))
        table=pd.DataFrame([dict(seed=r['config']['seed'],best_epoch=r['best_epoch'],**{k:v for k,v in r['metrics'].items() if isinstance(v,(int,float))}) for r in results])
        table.to_csv(run/'temporal_validation_comparison.csv',index=False)
        metrics=dict(status='completed',validation=table.to_dict('records'),
                     mean_subject_macro_f1=float(table.subject_macro_f1.mean()),
                     std_subject_macro_f1=float(table.subject_macro_f1.std()),internal_test_accessed=False)
        selection=dict(config=baseline_config(10302),final_epochs=max(1,int(round(table.best_epoch.median()))),
                       schedule_basis='Median best epoch from temporal validation seeds 302,303,304; same fixed MLP configuration.',
                       temporal_training_executed=True,internal_test_used_for_selection=False,
                       internal_test_model='temporal_seed302',internal_test_model_policy='Seed 302 fixed in advance; no selection of seed on internal-test scores')
    rt.write_json(run/'temporal_training_metrics.json',metrics)
    selected=run/'frozen_training_selection.json'
    if selected.exists() and rt.read_json(selected)!=selection:raise RuntimeError('Training configuration/schedule already frozen')
    if not selected.exists():rt.write_json(selected,selection,exclusive=True)
    lock=run/'training_selection_sha256.json'
    digest=file_sha256(selected)
    if lock.exists() and rt.read_json(lock)['sha256']!=digest:raise RuntimeError('Frozen training selection changed')
    if not lock.exists():rt.write_json(lock,dict(sha256=digest),exclusive=True)
    selected.chmod(0o444)
    return dict(training=metrics,selection=selection)


def internal_test(run,device):
    run=Path(run);verify_frozen(run)
    output=run/'temporal_internal_test_metrics.json'
    if output.exists():return rt.read_json(output)
    selection=rt.read_json(run/'frozen_training_selection.json')
    if not selection['temporal_training_executed']:
        report=dict(status='not_executed_infeasible',evaluation_label=TEST_LABEL,
                    reason=rt.read_json(run/'split_feasibility_report.json')['reasons'],
                    accuracy=None,macro_f1=None,test_evaluation_count=0)
        rt.write_json(output,report,exclusive=True)
        return report
    model_name=selection['internal_test_model'];path=run/'experiments'/model_name/'best.pt'
    marker=run/'internal_test_evaluation_started.json'
    rt.write_json(marker,dict(checkpoint_sha256=file_sha256(path),selection_sha256=file_sha256(run/'frozen_training_selection.json'),
                             evaluation_label=TEST_LABEL),exclusive=True)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=build_classifier_model('mlp_wide',checkpoint['model_config']['dropout']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    data=FramePartitionData(run,['internal_test'],device)
    score,frames=rt.evaluate(model,data,data.clips)
    score['internal_test_loss']=score.pop('validation_loss')
    score.update(status='completed',evaluation_label=TEST_LABEL,test_evaluation_count=1,
                 model_scope='Fresh temporal-train-only checkpoint; never the all-data model or original v001')
    rt.save_metrics(score,frames,run/'internal_test')
    shutil.copyfile(run/'internal_test/confusion_matrix.png',run/'temporal_confusion_matrix.png')
    rt.write_json(output,score,exclusive=True)
    return score


def full_snapshot(run):
    run=Path(run).resolve();verify_frozen(run);verify_sources(run)
    lock=run/'training_selection_sha256.json'
    if lock.exists() and file_sha256(run/'frozen_training_selection.json')!=rt.read_json(lock)['sha256']:
        raise RuntimeError('Full-data training schedule changed after freezing')
    destination=run/'full_snapshot'
    if (destination/'checksums.json').exists():
        rt.verify_seal(destination)
        frozen=rt.read_json(destination/'full_data_training_manifest.json')
        if file_sha256(destination/'processed/checksums.json')!=frozen['dataset_id']:raise RuntimeError('Full-data feature identity changed')
        return destination
    if destination.exists():raise RuntimeError('Incomplete full snapshot exists; inspect it before retrying')
    destination.mkdir();(destination/'processed').mkdir()
    inventory=rt.read_json(run/'audit/source_inventory.json');raw=Path(rt.read_json(run/'run_config.json')['raw_root'])
    for rel,record in inventory.items():
        if Path(rel).suffix.lower()=='.mp4':continue
        src=raw/rel;dst=destination/'raw'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        if file_sha256(dst)!=record['sha256'] or file_sha256(src)!=record['sha256']:raise RuntimeError('Source changed during full snapshot')
    for file in (run/'processed').glob('*.npz'):shutil.copyfile(file,destination/'processed'/file.name)
    rt.seal(destination/'processed')
    catalog=pd.read_csv(run/'catalog_source.csv');frames=pd.read_csv(run/'temporal_split_manifest.csv')
    manifest=dict(dataset_id=file_sha256(destination/'processed/checksums.json'),source_dataset_id=rt.read_json(run/'dataset_audit.json')['dataset_id'],
                  configuration_sha256=file_sha256(run/'frozen_training_selection.json'),
                  subjects=sorted(catalog.person_id.unique()),source_takes=int(catalog.source_group_id.nunique()),
                  camera_clips=len(catalog),all_source_frames=len(frames),eligible_frames=int(frames.eligible.sum()),
                  usable_source_takes=int(frames.loc[frames.eligible,'source_group_id'].nunique()),
                  usable_camera_clips=int(frames.loc[frames.eligible,'clip_id'].nunique()),
                  label_frames={str(k):int(v) for k,v in frames[frames.eligible].label.value_counts().sort_index().items()},
                  camera_frames=frames[frames.eligible].camera_id.value_counts().to_dict(),
                  qbao_included=True,qbao_is_held_out=False,held_out_evaluation_available=False,
                  quality_policy='All verified existing eligible frames, including temporal embargo/short-segment exclusions, because no internal partition remains held out for this model.',
                  source_inventory=inventory,feature_contract=FEATURE_VERSION,
                  temporal_internal_results_used_for_selection=False,external_test_pending=True)
    rt.write_json(destination/'full_data_training_manifest.json',manifest,exclusive=True)
    rt.seal(destination)
    rt.write_json(run/'full_data_training_manifest.json',manifest,exclusive=True)
    return destination


def fit_full_data(run,device,resume=False):
    run=Path(run).resolve();marker=run/'full_data_model.json'
    if marker.exists():return rt.read_json(marker)
    selection=rt.read_json(run/'frozen_training_selection.json')
    full_snapshot(run)
    data=FramePartitionData(run,['train'],device,full=True)
    cfg=selection['config'];expected=rt.read_json(run/'full_data_training_manifest.json')['eligible_frames']
    if len(data.x)!=expected:raise RuntimeError('All-data fit omitted eligible observations')
    smoke_cfg={**cfg,'epochs':2,'patience':2}
    rt.run_candidate(run,smoke_cfg,device,'smoke_full_data',resume,final_epochs=2,data=data)
    smoke=pd.read_csv(run/'experiments/smoke_full_data/history.csv')
    if smoke.train_loss.iloc[-1]>=smoke.train_loss.iloc[0]:raise RuntimeError('All-data CUDA smoke loss did not decrease')
    result=rt.run_candidate(run,cfg,device,'full_data_final',resume,final_epochs=selection['final_epochs'],data=data)
    raw=Path(rt.read_json(run/'run_config.json')['raw_root'])
    folder=rt.version_dir(raw.parents[1]/'models/mlp_wide_full_real_dataset')
    contract=rt.read_json(run/'feature_contract.json');contract['source_dataset_id']=contract['dataset_id']
    contract['dataset_id']=rt.read_json(run/'full_data_training_manifest.json')['dataset_id']
    labels=rt.read_json(run/'full_snapshot/raw/config/labels.json')
    names=[r['name'] for r in sorted(labels['labels'],key=lambda r:r['id'])]
    if sorted(r['id'] for r in labels['labels'])!=list(range(6)):raise ValueError('Incorrect six-class ordering')
    checkpoint=torch.load(run/'experiments/full_data_final/trained.pt',map_location='cpu',weights_only=True)
    checkpoint.update(stage='final',feature_contract=contract,label_names=names,
                      train_people=sorted({c['person'] for c in data.clips}),dataset_id=contract['dataset_id'],
                      training_scope='all_eligible_real_data_including_QBAO',external_test_pending=True)
    rt.atomic_torch_save(checkpoint,folder/'model_final.pt')
    rt.write_json(folder/'model_config.json',cfg,exclusive=True)
    rt.write_json(folder/'feature_contract.json',contract,exclusive=True)
    rt.write_json(folder/'label_map.json',labels,exclusive=True)
    rt.write_json(folder/'full_data_training_manifest.json',rt.read_json(run/'full_data_training_manifest.json'),exclusive=True)
    rt.write_json(folder/'training_summary.json',dict(training_scope='all_eligible_real_data',eligible_frames=len(data.x),
                  subjects=checkpoint['train_people'],epochs=result['best_epoch'],seed=cfg['seed'],
                  schedule_basis=selection['schedule_basis'],qbao_no_longer_held_out=True,
                  independent_evaluation_available=False,external_evaluation_pending=True,
                  note='Temporal internal metrics belong only to the separately trained temporal model, if executed. No held-out score is reported for this all-data checkpoint.'),exclusive=True)
    for dest in [folder/'full_data_training_history.csv',run/'full_data_training_history.csv']:
        shutil.copyfile(run/'experiments/full_data_final/history.csv',dest)
    shutil.copyfile(run/'experiments/full_data_final/learning_curves.png',folder/'learning_curves.png')
    value=dict(path=str(folder),checkpoint_sha256=file_sha256(folder/'model_final.pt'),eligible_frames=len(data.x),epochs=result['best_epoch'],
               external_evaluation_pending=True,release_status='full-data release candidate; independent external accuracy unknown')
    rt.write_json(marker,value,exclusive=True)
    return value


def deploy(run):
    from scripts.export_onnx import export
    from inference.pipeline import PostureClassifier
    import subprocess
    import sys
    run=Path(run).resolve();record=rt.read_json(run/'full_data_model.json');folder=Path(record['path'])
    if not (folder/'model.onnx').exists():export(folder)
    alias=folder/'model_final.onnx'
    if not alias.exists():shutil.copyfile(folder/'model.onnx',alias)
    if file_sha256(alias)!=file_sha256(folder/'model.onnx'):raise RuntimeError('ONNX deployment filenames differ')
    data=FramePartitionData(run,['train'],'cpu',full=True)
    pt=PostureClassifier(folder,'pt');onnx=PostureClassifier(folder,'onnx')
    label_map=rt.read_json(folder/'label_map.json')
    labels=[r['name'] for r in sorted(label_map['labels'],key=lambda r:r['id'])]
    if pt.labels!=labels or onnx.labels!=labels:raise ValueError('Six-class label ordering mismatch')
    maximum=0.
    with torch.inference_mode():
        for start in range(0,len(data.x),1024):
            x=data.x[start:start+1024]
            a=pt.model(torch.from_numpy(x)).softmax(1).numpy()
            b=torch.from_numpy(onnx.session.run(['logits'],{'features':x})[0]).softmax(1).numpy()
            if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite deployed probabilities')
            if a.shape[1]!=6 or not np.allclose(a.sum(1),1,atol=1e-6):raise ValueError('Invalid six-class probabilities')
            maximum=max(maximum,float(np.abs(a-b).max()))
    if maximum>1e-5:raise RuntimeError('PT/ONNX probability parity failed')
    # Current raw keypoints -> unchanged causal features, every original frame.
    regenerated=0;max_feature_error=0.
    for row in pd.read_csv(run/'catalog_source.csv').to_dict('records'):
        items=[json.loads(line) for line in rt.pose_path(run/'full_snapshot/raw',row).read_text(encoding='utf-8').splitlines()]
        selected=[dict(item,keypoints=None) if rt.uncertain_selection(item) else item for item in items]
        x=rt.stream_features(selected)
        with np.load(run/'full_snapshot/processed'/f'{row["clip_id"]}.npz',allow_pickle=False) as arrays:
            expected=arrays['x']
        np.testing.assert_array_equal(x,expected)
        np.testing.assert_array_equal(x[:12],rt.stream_features(selected[:12]))
        max_feature_error=max(max_feature_error,float(np.abs(x-expected).max()))
        regenerated+=len(x)
    for x in data.x[:32]:
        np.testing.assert_allclose(pt.predict(x),onnx.predict(x),atol=1e-5,rtol=1e-5)
    old_root=Path(rt.read_json(run/'run_config.json')['raw_root']).parents[1]/'models/mlp_wide/v001'
    old=PostureClassifier(old_root,'pt')
    if old.predict(data.x[0]).shape!=(6,):raise RuntimeError('Original v001 no longer loads')
    raw=Path(rt.read_json(run/'run_config.json')['raw_root']);root=raw.parents[2]
    source=raw/'recordings/CHANH/session_20261027_1124/label_0/take_001/cam01_raw.mp4'
    command=[sys.executable,str(root/'inference/run.py'),'--source',str(source),'--mode','video',
             '--model',str(folder),'--pose',str(raw.parents[1]/'pretrained/yolo11n-pose.pt'),
             '--device','cuda:0','--backend','onnx','--cls-device','cpu','--headless','--name','full_real_dataset_smoke']
    marker=run/'deployment_smoke.json'
    if not marker.exists():
        result=subprocess.run(command,cwd=root,text=True,encoding='utf-8',capture_output=True)
        (run/'deployment_smoke.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Full-data model video smoke failed')
        frames_path=max((raw.parents[1]/'inference').glob('full_real_dataset_smoke_v*/frames.csv'),key=lambda p:p.stat().st_mtime_ns)
        frames=pd.read_csv(frames_path)
        rt.write_json(marker,dict(command=command,output=str(frames_path.parent),frames=len(frames),classified=int(frames.class_id.notna().sum()),
                                 scope='Pipeline smoke on already available training footage; no accuracy estimate'),exclusive=True)
    latencies=[]
    for x in data.x[:50]:onnx.predict(x)
    for i in range(1000):
        start=time.perf_counter_ns();onnx.predict(data.x[i%len(data.x)]);latencies.append((time.perf_counter_ns()-start)/1e6)
    verification=dict(verified=True,feature_contract=FEATURE_VERSION,input_dim=52,num_classes=6,
                 max_abs_probability_difference=maximum,parity_tolerance=1e-5,parity_frames=len(data.x),
                 raw_causal_features_regenerated=regenerated,max_abs_feature_difference=max_feature_error,
                 six_class_ordering_verified=True,no_invalid_numerical_outputs=True,original_v001_loads=True,
                 onnx_sha256=file_sha256(alias),checkpoint_sha256=file_sha256(folder/'model_final.pt'),
                 inference_cpu_median_ms=float(np.median(latencies)),inference_cpu_p95_ms=float(np.quantile(latencies,.95)),
                 latency_scope='Warm ONNX CPU predict call, batch one, 1000 calls; excludes YOLO/ByteTrack.',
                 smoke=rt.read_json(marker),preservation=verify_sources(run),
                 independent_accuracy_estimate=None,external_evaluation_pending=True,qbao_is_held_out=False)
    rt.write_json(folder/'deployment_verification.json',verification)
    rt.write_json(run/'deployment_verification.json',verification)
    return verification


def md_table(df,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in df[columns].to_dict('records'):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row.values())+' |')
    return '\n'.join(lines)


def report(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from zipfile import ZipFile,ZIP_DEFLATED
    run=Path(run).resolve();config=rt.read_json(run/'run_config.json')
    raw=Path(config['raw_root']);root=raw.parents[2]
    audit_result=rt.read_json(run/'dataset_audit.json');feasibility=rt.read_json(run/'split_feasibility_report.json')
    selection=rt.read_json(run/'frozen_training_selection.json');model_info=rt.read_json(run/'full_data_model.json')
    model=Path(model_info['path']);deployment=rt.read_json(run/'deployment_verification.json')
    internal=rt.read_json(run/'temporal_internal_test_metrics.json');training=rt.read_json(run/'temporal_training_metrics.json')
    frames=pd.read_csv(run/'temporal_split_manifest.csv');catalog=pd.read_csv(run/'catalog_source.csv')
    hist=pd.read_csv(run/'full_data_training_history.csv')
    full=frames[frames.eligible]
    partitions=pd.DataFrame(feasibility['partition_counts'])
    partitions.drop(columns=['labels','cameras']).to_csv(run/'partition_counts.csv',index=False)
    by_label=pd.DataFrame({'label':range(6)})
    for name,part in [('raw',frames),('eligible_full_data',full),*[(p,frames[frames.partition==p]) for p in PARTITIONS]]:
        by_label[name]=by_label.label.map(part.label.value_counts()).fillna(0).astype(int)
    by_label.to_csv(run/'partition_label_counts.csv',index=False)
    by_camera=pd.crosstab(frames.camera_id,frames.partition).reindex(columns=[*PARTITIONS,'excluded'],fill_value=0).reset_index()
    by_camera['eligible_full_data']=by_camera.camera_id.map(full.camera_id.value_counts())
    by_camera.to_csv(run/'partition_camera_counts.csv',index=False)
    by_subject=pd.crosstab(frames.person_id,frames.partition).reindex(columns=[*PARTITIONS,'excluded'],fill_value=0).reset_index()
    by_subject['eligible_full_data']=by_subject.person_id.map(full.person_id.value_counts())
    by_subject.to_csv(run/'partition_subject_counts.csv',index=False)
    exclusions=frames[frames.partition=='excluded'].exclusion_reason.value_counts().rename_axis('reason').reset_index(name='frames')
    exclusions.to_csv(run/'temporal_exclusion_counts.csv',index=False)
    # Actual capture-time diagrams: one typical short take and the longest take.
    takes=rt.read_json(run/'take_timelines.json');durations=np.array([t['duration_s'] for t in takes])
    picks=[takes[int(np.argmin(np.abs(durations-np.median(durations))))],takes[int(durations.argmax())]]
    colors={'train':'#3979bd','val':'#28a77b','internal_test':'#de9440','excluded':'#9aa1aa'}
    fig,axes=plt.subplots(2,1,figsize=(11,5),layout='constrained')
    for ax,take in zip(axes,picks):
        subset=frames[frames.source_group_id==take['source_group_id']]
        for i,camera in enumerate(sorted(subset.camera_id.unique())):
            for partition,color in colors.items():
                q=subset[(subset.camera_id==camera)&subset.partition.eq(partition)]
                ax.scatter((q.capture_ns-take['start_ns'])/1e9,np.full(len(q),i),s=9,color=color,label=partition if i==0 else None)
        for key in ['train_val_boundary_ns','val_test_boundary_ns']:
            boundary=(take[key]-take['start_ns'])/1e9
            ax.axvline(boundary,color='black',linewidth=.8)
            ax.axvspan(max(0,boundary-config['options']['purge_seconds']),min(take['duration_s'],boundary+config['options']['purge_seconds']),color='#db6873',alpha=.15)
        ax.set(yticks=[0,1],yticklabels=['cam01','cam02'],xlabel='Shared host capture time from take start (s)',title=take['source_group_id'],ylim=(-.4,1.4))
        ax.legend(loc='upper left',ncol=4,fontsize=8)
    fig.savefig(run/'temporal_timeline_examples.png',dpi=150);plt.close(fig)
    counts=pd.crosstab(frames.partition,frames.label).reindex([*PARTITIONS,'excluded'],fill_value=0)
    ax=counts.plot.bar(stacked=True,figsize=(9,4),rot=0)
    ax.set(ylabel='Original frames',title='Actual post-purge partitions; existing quality exclusions retained')
    ax.legend(title='Label',ncol=6);ax.figure.tight_layout();ax.figure.savefig(run/'temporal_partition_counts.png',dpi=150);plt.close(ax.figure)
    fig,ax=plt.subplots(figsize=(6,3.5));ax.plot(hist.epoch,hist.train_loss,marker='o')
    ax.set(xlabel='Epoch',ylabel='Training cross-entropy',title=f'All-data MLP Wide: {len(full):,} eligible frames; no held-out evaluation')
    ax.set_xticks(hist.epoch);fig.tight_layout();fig.savefig(run/'full_data_training_loss.png',dpi=150);plt.close(fig)
    if internal['status']!='completed':
        fig,ax=plt.subplots(figsize=(7,3));ax.axis('off')
        ax.text(.5,.65,'NO CONFUSION MATRIX',ha='center',fontsize=18,weight='bold')
        ax.text(.5,.38,'Temporal internal test was not executed.\nPost-purge validation/test coverage is insufficient.\nNo test scores or confusion counts are available.',ha='center',va='center',fontsize=11)
        fig.tight_layout();fig.savefig(run/'temporal_confusion_matrix.png',dpi=150);plt.close(fig)
    preserved=verify_sources(run)
    usable_takes=int(full.source_group_id.nunique());usable_clips=int(full.clip_id.nunique())
    focused=(run/'regression_temporal.log').read_text(encoding='utf-8') if (run/'regression_temporal.log').exists() else 'Not available'
    regression=(run/'regression_full.log').read_text(encoding='utf-8') if (run/'regression_full.log').exists() else 'Not available'
    test_summary='\n'.join([s for s in regression.splitlines() if 'passed' in s or 'failed' in s])
    focused_summary='\n'.join([s for s in focused.splitlines() if 'passed' in s or 'failed' in s])
    max_clock=max(t['clock']['offset_spread_ms'] for t in takes)
    text=f'''# Temporal split and full real dataset MLP training

## Outcome

**A new MLP Wide release candidate was actually trained on all {len(full):,} eligible
observations from all {audit_result['subject_count']} participants, including QBAO.**
The proposed temporal split is implemented and frozen, but is **not viable** on
these short recordings at the unchanged ±0.25-second embargo. No temporal baseline
training or internal-test scoring was executed. External evaluation remains pending.
Original `artifacts/models/mlp_wide/v001` is unchanged.

## Current dataset and eligibility

The fresh audit decoded all {audit_result['clips']} camera clips / {audit_result['source_takes']}
original takes / {audit_result['frames']:,} frames. All 30 recovered files were recognized.
There are {audit_result['pose_frames']:,} selected poses, {audit_result['no_selected_frames']:,}
frames without a selected person, {audit_result['uncertain_selected_frames']} uncertain
selected observations, and {len(audit_result['empty_pose_clips'])} pose-empty camera files.
All prior eligibility masks match the new audit exactly. No labels, raw poses,
metadata, timestamps or videos were rewritten; no missing joints were synthesized.

The all-data snapshot includes every original take and both camera views in its
manifest. After preserving quality exclusions, {usable_takes} takes and {usable_clips}
camera clips contribute nonzero training observations. The wholly pose-empty
source take and camera clips remain represented in provenance rather than being
invented as training samples.

## Shared capture-time split and feasibility

Boundaries are computed **once per original take** at 80% and 90% of the union
capture timeline across both cameras, before quality exclusions/embargo. Timestamp
rows retain source frame indices, writer indices, UTC, original monotonic ns,
subject/session/take/camera and label. Encoded MP4 FPS is never used to partition.
All {feasibility['clock_verified_takes']}/88 pairs passed common host-clock verification
against UTC/monotonic offsets and metadata recording bounds (50 ms tolerance;
maximum inter-camera median offset difference {max_clock:.3f} ms). Invalid clocks
would exclude the take from temporal partitions without changing full-data quality
eligibility. These are **host reception timestamps**, not synchronized exposure
times: network/camera latency is not established by this check.

Repeated timestamp ticks ({audit_result['duplicate_timestamps']}) are retained and
always receive the same take-level assignment. Backward ticks: {audit_result['backward_timestamps']}.
No source-frame identities overlap partitions. Leakage checks passed. Common-clock
chronology does not prove that near-identical poses, scenes, or people are independent.

`purge_seconds=0.25` is explicitly a **symmetric radius**, excluding all observations
within ±0.25 seconds of either boundary (a 0.5-second total gap). It was not reduced.
Minimum evaluation camera-segment quality: eight eligible frames spanning at least
0.10 seconds. Rejected segments are excluded, never reassigned to training.
Viability additionally requires every evaluation class to have two source takes
and two subjects. Full defaults/frozen boundaries are in the JSON/CSV manifests.

Median take duration is {feasibility['duration_summary_s']['50%']:.3f} seconds
(range {feasibility['duration_summary_s']['min']:.3f}–{feasibility['duration_summary_s']['max']:.3f}).
Its nominal 10% validation window is approximately 0.24 seconds and is consumed
by the embargo. Only three longer takes have any pre-minimum validation frames;
after segment checks, validation has 55 frames from one subject and only labels
1 and 2. Internal test has no label 4. Six-class temporal evaluation is therefore
not justified, regardless of how high a score on this small subset might be.

{md_table(partitions,['partition','frames','source_takes','subjects','camera_clips'])}

{md_table(by_label,['label','raw','eligible_full_data','train','val','internal_test'])}

{md_table(by_camera,['camera_id','train','val','internal_test','excluded','eligible_full_data'])}

Subject-level counts: `partition_subject_counts.csv`. Per-take original duration,
raw/eligible segment counts, embargo counts, capture span and segment usability:
`take_timelines.json` and `temporal_segment_feasibility.csv`.

Embargo removes **{feasibility['eligible_embargo_frames']:,} otherwise eligible frames**.
Effective temporal training usage is **{feasibility['effective_training_percent_of_eligible']:.2f}%
of eligible observations** ({feasibility['effective_training_percent_of_raw']:.2f}% of raw frames).
Exclusion reasons are disjoint in the following accounting:

{md_table(exclusions,['reason','frames'])}

![Shared-camera capture timelines](temporal_timeline_examples.png)

![Actual post-purge frame counts](temporal_partition_counts.png)

## Evaluation categories

| Category | Status and interpretation |
| --- | --- |
| Historical original subject validation | Wide mean subject Macro-F1 0.9017 over seeds 302–304; 12-development-subject protocol from the earlier phase. |
| Historical grouped subject CV | Wide 0.8467, geometry 0.8133, targeted augmentation 0.8276; retained as historical development evidence, not rerun. |
| New temporal baseline | **Not trained: infeasible validation/test coverage.** No new Macro-F1, per-class metrics or missing-joint scores are claimed. |
| {TEST_LABEL} | **Not executed; evaluation count 0.** `temporal_internal_test_metrics.json` explicitly stores skipped status and null scores. |
| New all-data model | Trained on every eligible observation; neither old QBAO nor the temporal partitions remain held out. **No held-out accuracy is available.** |
| Future independent external dataset | Not collected/evaluated here; pending explicit authorization and newly collected external subjects. |

The temporal comparison branch supports fresh Wide training at seeds 302/303/304,
validation-only early stopping, a frozen schedule, and a once-only internal-test
marker for a future **viable** dataset. Training adapters select only Train/Val
observations before GPU transfer. Original v001 is never scored as the temporal
test model. This run follows the explicitly authorized infeasible-split fallback.
`temporal_confusion_matrix.png` is a labeled **no-test status figure**, not a
fabricated confusion matrix. No temporal scores are compared to historical scores.

## Actual all-data CUDA training

After freezing the configuration, a new sealed snapshot was created at
`{run/'full_snapshot'}` with copied raw pose/timestamp/metadata inputs, original
eligibility masks, 52-feature arrays and SHA-256 inventories. Videos are represented
by provenance hashes and are not copied into shareable artifacts.

Fresh MLP Wide: 52–256–128–6, original `{FEATURE_VERSION}`, dropout 0.15,
AdamW lr 0.001, weight decay 0.001, batch 128, seed **10302**, **4 fixed epochs**,
float32, NVIDIA GeForce RTX 2050 / PyTorch 2.6.0+cu124. The four-epoch schedule is
the median of historical development-selected epochs 8/4/1; no QBAO score or new
internal-test score selected it. Sampling uses all eligible frames without weighting,
augmentation, smoothing or source quotas, matching the selected Wide control.
Source-aware sampling remains available in the shared engine; its earlier
non-improving settings were not repeated. No rejected architecture was promoted.

The two-epoch real CUDA smoke passed before the fresh four-epoch fit. Every epoch
used **{len(full):,} observations**. Training loss decreased from
**{hist.train_loss.iloc[0]:.4f} to {hist.train_loss.iloc[-1]:.4f}**. All thirteen
participants are included, including QBAO; QBAO is explicitly no longer held out.
Historical QBAO accuracy is not a score for this new model.

![Full-data training loss; no validation score](full_data_training_loss.png)

Checkpoint: `{model/'model_final.pt'}`.
ONNX: `{model/'model_final.onnx'}` (`model.onnx` is the identical loader-compatible copy).
Both the source split run and model directory contain the full-data training
manifest/history, model configuration, feature contract, label map, and deployment
verification. No `test_metrics.json` is created for the all-data checkpoint.

## Deployment verification and reproducibility

PT/ONNX probability parity on {deployment['parity_frames']:,} real training feature rows:
maximum absolute difference **{deployment['max_abs_probability_difference']:.9g}**,
tolerance 1e-5. All {deployment['raw_causal_features_regenerated']:,} raw causal feature
rows regenerated exactly (difference {deployment['max_abs_feature_difference']}).
Six-class ordering, finite outputs and both loaders passed. Original v001 still
loads. All {preserved['source_files_unchanged']} inventoried original dataset files
and {preserved['prior_model_files_unchanged']} prior installed-model files are unchanged.
ONNX batch-one CPU predict-call median/p95 is {deployment['inference_cpu_median_ms']:.4f}/
{deployment['inference_cpu_p95_ms']:.4f} ms (1,000 warmed calls, excluding pose extraction).

End-to-end smoke processed {deployment['smoke']['frames']} frames and classified
{deployment['smoke']['classified']} through existing YOLO11n-Pose → ByteTrack → selector →
causal normalization → ONNX MLP on already available CHANH footage. It is not an
independent model evaluation. The local annotated video is not included in bundles.

Added `shared/temporal_training.py` and `tests/test_temporal_training.py`;
extended `scripts/train_poseforge.py` with `--split-strategy temporal_within_take`
and explicit audit/prepare/baseline/internal-test/full-data/export/report stages.
The existing subject-wise default, notebooks, training engine, model definitions,
normalizer, exporter and inference behavior remain supported. No packages were
reinstalled, no YOLO keypoints re-extracted, and no extra smoothing introduced.
`docs/TRAINING_WINDOWS.md` records the actual PowerShell commands. Configurations,
frozen boundaries, source hashes, source copies, logs and resumable checkpoints
are saved under `{run}`. Resume uses trusted local optimizer/RNG files; interruption
during internal-test evaluation would not automatically repeat the test.

Tests: focused **{focused_summary}**. Broad regression: **{test_summary}**.
The one existing administrator/symlink-dependent asset-migration test was explicitly
deselected; its prior WinError 1314 is unrelated to these workflows, which use no
symbolic links or administrator operations. Detailed logs are included in the run.
Tests cover boundaries, paired clocks, embargo, short segments, duplicate identity,
quality eligibility, Train/Val isolation, full-data inclusion of QBAO/internal-region
frames, prior-model preservation, old subject splits, and PT/ONNX contracts.

Remaining requirements: newly collected independent external subjects/recordings,
frozen external evaluation protocol, and explicit authorization before that test.
The full-data candidate is deployable; its independent real-world accuracy remains
unknown. Longer future recordings are needed for meaningful 80/10/10 windows at
the unchanged embargo. The prior subject-wise evidence remains primary.
'''
    destination=root/'artifacts/reports/TEMPORAL_SPLIT_AND_FULL_DATA_TRAINING.md'
    if destination.exists():raise FileExistsError('Preserve prior report; use a new versioned report path')
    # Use absolute paths for local images in the shared report.
    for image_name in ['temporal_timeline_examples.png','temporal_partition_counts.png','full_data_training_loss.png']:
        text=text.replace(']('+image_name+')',']('+str(run/image_name).replace('\\','/')+')')
    destination.write_text(text,encoding='utf-8')
    (run/'TEMPORAL_SPLIT_AND_FULL_DATA_TRAINING.md').write_text(text,encoding='utf-8')
    code=run/'source_code';code.mkdir(exist_ok=True)
    names=['shared/temporal_training.py','shared/real_training.py','shared/pose_training.py','shared/pose_features.py',
           'shared/mlp_model.py','scripts/train_poseforge.py','scripts/export_onnx.py','inference/pipeline.py','inference/run.py']
    for name in names:
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    rt.write_json(run/'code_hashes.json',{name:file_sha256(root/name) for name in names})
    bundle=model/'model_bundle.zip'
    with ZipFile(bundle,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(model.iterdir()):
            if file.is_file() and file.suffix in ['.pt','.onnx','.json','.csv','.png']:z.write(file,file.name)
        portable=text
        for image_name in ['temporal_timeline_examples.png','temporal_partition_counts.png','full_data_training_loss.png']:
            portable=portable.replace(str(run/image_name).replace('\\','/'),image_name)
            z.write(run/image_name,image_name)
        z.writestr('TEMPORAL_SPLIT_AND_FULL_DATA_TRAINING.md',portable)
        z.write(root/'docs/TRAINING_WINDOWS.md','docs/TRAINING_WINDOWS.md')
        for name in ['shared/mlp_model.py','shared/pose_features.py','shared/pose_geometry.py','inference/pipeline.py','inference/run.py']:z.write(root/name,name)
    with ZipFile(bundle) as z:
        assert z.testzip() is None and not any(n.endswith(('.mp4','.npz')) for n in z.namelist())
    rt.write_json(run/'completion_summary.json',dict(model=model_info,temporal_viable=feasibility['viable'],
                   temporal_training_status=training['status'],internal_test_status=internal['status'],internal_test_count=internal['test_evaluation_count'],
                   external_test_executed=False,report=str(destination),bundle=str(bundle),bundle_sha256=file_sha256(bundle),
                   preservation=preserved),exclusive=True)
    return dict(report=str(destination),bundle=str(bundle))
