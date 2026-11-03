"""Frozen source-take allocation, real causal GRU windows and matched controls."""
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
from shared.mlp_model import build_classifier_model
from shared.pose_features import FEATURE_VERSION
from shared.sequence_data import sequence_contract,window_records

LOG=logging.getLogger('poseforge.training')
TEST_LABEL='INTERNAL TAKE-LEVEL TEST – SEEN SUBJECTS, UNSEEN TAKES'
PARTS=('train','val','test')
MODELS=('mlp_control','gru16','gru32')


def load_clip(run,row):
    run=Path(run)
    with np.load(run/'processed'/f'{row["clip_id"]}.npz',allow_pickle=False) as z:
        x=z['x'].reshape(-1,52).copy();eligible=z['eligible'].copy();ticks=z['capture_ns'].copy()
    source=Path(rt.read_json(run/'run_config.json')['source_run'])/'snapshot'
    items=[json.loads(line) for line in rt.pose_path(source,row).read_text(encoding='utf-8').splitlines()]
    tracks=[item.get('selected_track_id',item.get('track_id')) for item in items]
    sessions=[item.get('track_session') for item in items]
    return x,eligible,ticks,tracks,sessions


def allocate_takes(groups,seed=302,candidates=512):
    """Rank deterministic allocations by predefined coverage/quality rules only."""
    groups=groups.copy()
    included=groups[groups.eligible_frames>0]
    excluded=groups[groups.eligible_frames==0].source_group_id.tolist()
    if set(included.label)!=set(range(6)):raise ValueError('Six usable labels required')
    if included.source_group_id.duplicated().any():raise ValueError('Duplicate source take identity')
    rng=np.random.default_rng(seed);people=set(included.person_id)
    best=None;best_assignment=None;ranking=[]
    for candidate in range(candidates):
        assignment={group:'excluded' for group in excluded}
        for label,piece in included.groupby('label',sort=True):
            ids=piece.source_group_id.to_numpy().copy();rng.shuffle(ids)
            if len(ids)<3:raise ValueError('Not enough takes for six-class stratification')
            n=max(1,int(round(len(ids)*.15)))
            if 2*n>=len(ids):n=1
            for i,group in enumerate(ids):assignment[group]='val' if i<n else 'test' if i<2*n else 'train'
        table=included.assign(partition=included.source_group_id.map(assignment))
        train=table[table.partition=='train'];hard=len(people-set(train.person_id))*1000
        diversity=0;imbalance=0.
        for part in PARTS:
            q=table[table.partition==part]
            hard+=1000*(6-q.label.nunique())
            hard+=1000*sum(not (q.loc[q.label==label,'valid32_targets']>0).any() for label in range(6))
            if part!='train':diversity+=q.person_id.nunique()
            for label in range(6):
                global_rate=included.loc[included.label==label,'eligible_frames'].sum()/included.eligible_frames.sum()
                local_rate=q.loc[q.label==label,'eligible_frames'].sum()/max(1,q.eligible_frames.sum())
                imbalance+=(local_rate-global_rate)**2
        key=(hard,-diversity,float(imbalance),candidate)
        ranking.append(dict(candidate=candidate,hard_penalty=hard,validation_test_subject_diversity=diversity,label_frame_imbalance=imbalance))
        if best is None or key<best:best=key;best_assignment=assignment
    if best[0]:raise ValueError('Predefined split coverage constraints infeasible')
    return best_assignment,dict(seed=seed,candidate_count=candidates,chosen_candidate=int(best[3]),
               ranking='hard label/train-subject/GRU32 coverage, maximum validation+test subject diversity, minimum eligible-label imbalance, candidate index',
               excluded_groups=excluded,chosen_score=list(best),model_scores_used=False),pd.DataFrame(ranking)


def check_leakage(catalog):
    if catalog.clip_id.duplicated().any():raise ValueError('Duplicate camera clip')
    if catalog.groupby('source_group_id').partition.nunique().max()!=1:raise ValueError('One source take spans partitions/cameras')
    if not set(catalog.label)<=set(range(6)):raise ValueError('Invalid class')
    for part in PARTS:
        q=catalog[(catalog.partition==part)&(catalog.usable_frames>0)]
        if set(q.label)!=set(range(6)):raise ValueError('Missing partition class coverage')
    train_people=set(catalog.loc[catalog.partition=='train','person_id'])
    if not set(catalog.loc[catalog.partition.isin(['val','test']),'person_id'])<=train_people:
        raise ValueError('Internal take test must use training-seen subjects')
    return dict(passed=True,source_take_overlap=False,camera_pair_leakage=False,all_six_labels_per_partition=True,
                train_subject_coverage=sorted(train_people),test_subjects_seen_in_training=True,frame_random_split=False,
                temporal_embargo_used=False,group_key='person/session/label/original source_take_id')


def audit_split(raw_root,source_run,output,options):
    run=rt.version_dir(output);raw_root=Path(raw_root).resolve();source_run=Path(source_run).resolve()
    result=rt.audit(raw_root,run/'audit',raw_root.parents[1]/'reextract_backups/20261030T115911Z_cc9dbeb9/manifest.json')
    inventory=rt.read_json(run/'audit/source_inventory.json');old=rt.read_json(source_run/'snapshot/source_inventory.json')
    for rel,value in old.items():
        if rel not in inventory or inventory[rel]['sha256']!=value['sha256']:raise RuntimeError('Original eligibility source changed: '+rel)
    catalog=pd.read_csv(run/'audit/clip_quality.csv');old_catalog=pd.read_csv(source_run/'catalog.csv')
    if set(catalog.clip_id)!=set(old_catalog.clip_id):raise RuntimeError('Reconcile curation for changed source catalog')
    (run/'processed').mkdir();(run/'splits').mkdir();(run/'sequences').mkdir()
    rt.write_json(run/'run_config.json',dict(source_run=str(source_run),raw_root=str(raw_root),options=options),exclusive=True)
    hashes=rt.read_json(source_run/'processed/checksums.json');quality=[]
    for row in catalog.to_dict('records'):
        src=source_run/'processed'/f'{row["clip_id"]}.npz'
        if file_sha256(src)!=hashes[src.name]:raise RuntimeError('Prior immutable features changed')
        shutil.copyfile(src,run/'processed'/src.name)
        x,eligible,ticks,tracks,sessions=load_clip(run,row)
        if eligible.sum()!=row['usable_frames']:raise RuntimeError('Eligibility changed')
        counts={}
        for length in [16,32]:
            records,resets=window_records(ticks,eligible,tracks,sessions,sequence_contract(length,**options['sequence']),1)
            counts[f'valid{length}_targets']=sum(r['accepted'] for r in records)
        quality.append(dict(clip_id=row['clip_id'],source_group_id=row['source_group_id'],person_id=row['person_id'],
                    label=row['label'],camera_id=row['camera_id'],frames=len(x),eligible_frames=int(eligible.sum()),
                    duplicate_ticks=int((np.diff(ticks)==0).sum()),backward_ticks=int((np.diff(ticks)<0).sum()),
                    large_gaps=int((np.diff(ticks)>.25e9).sum()),**counts))
    q=pd.DataFrame(quality);q.to_csv(run/'audit/sequence_feasibility.csv',index=False)
    groups=q.groupby('source_group_id').agg(person_id=('person_id','first'),label=('label','first'),
                     eligible_frames=('eligible_frames','sum'),valid32_targets=('valid32_targets','sum')).reset_index()
    assignments,allocation,ranking=allocate_takes(groups,options['seed'],options['split_candidates'])
    catalog['partition']=catalog.source_group_id.map(assignments)
    checks=check_leakage(catalog)
    catalog.to_csv(run/'split_manifest.csv',index=False);catalog.to_csv(run/'splits/split_manifest.csv',index=False)
    groups.assign(partition=groups.source_group_id.map(assignments)).to_csv(run/'audit/take_quality.csv',index=False)
    ranking.to_csv(run/'splits/allocation_candidates.csv',index=False)
    manifest=dict(strategy='take_level',allocation=allocation,partitions={part:sorted(g for g,p in assignments.items() if p==part) for part in [*PARTS,'excluded']},
                  dataset_id=result['dataset_id'],options=options,internal_test_label=TEST_LABEL,
                  test_policy='Frozen validation-only selection; one prediction pass for representative seed302 winner. Never score old all-data checkpoints.')
    rt.write_json(run/'split_manifest.json',manifest,exclusive=True);rt.write_json(run/'splits/split_manifest.json',manifest,exclusive=True)
    rt.write_json(run/'leakage_checks.json',checks,exclusive=True)
    summary={}
    for part in [*PARTS,'excluded']:
        p=catalog[catalog.partition==part]
        summary[part]=dict(takes=int(p.source_group_id.nunique()),camera_clips=len(p),subjects=int(p.person_id.nunique()),
                  eligible_frames=int(p.usable_frames.sum()),raw_frames=int(p.frames.sum()),
                  label_takes={str(k):int(v) for k,v in p.drop_duplicates('source_group_id').label.value_counts().sort_index().items()},
                  camera_eligible_frames={str(k):int(v) for k,v in p.groupby('camera_id').usable_frames.sum().items()})
    rt.write_json(run/'split_quality.json',dict(counts=summary,allocation=allocation,pose_empty_clips=catalog.loc[catalog.pose17_frames==0,'clip_id'].tolist(),
                   insufficient_gru32_clips=q.loc[q.valid32_targets==0,'clip_id'].tolist(),
                   excluded_reason='Only original groups with zero eligible observations are excluded from all partitions.'),exclusive=True)
    rt.seal(run/'processed')
    rt.write_json(run/'feature_contract.json',rt.read_json(source_run/'feature_contract.json'),exclusive=True)
    preservation={str(p.resolve()):file_sha256(p) for p in (raw_root.parents[1]/'models').rglob('*') if p.is_file()}
    rt.write_json(run/'prior_model_hashes.json',preservation,exclusive=True)
    names=['run_config.json','split_manifest.csv','split_manifest.json','feature_contract.json','audit/source_inventory.json']
    rt.write_json(run/'frozen_checksums.json',{n:file_sha256(run/n) for n in names},exclusive=True)
    for n in names:(run/n).chmod(0o444)
    LOG.info('Frozen take split %s: %s',run,summary)
    return run


def verify_run(run,sources=False):
    run=Path(run)
    for rel,digest in rt.read_json(run/'frozen_checksums.json').items():
        if file_sha256(run/rel)!=digest:raise RuntimeError('Frozen take split changed')
    if sources:
        raw=Path(rt.read_json(run/'run_config.json')['raw_root'])
        for rel,value in rt.read_json(run/'audit/source_inventory.json').items():
            if file_sha256(raw/rel)!=value['sha256']:raise RuntimeError('Original source changed')
        for path,digest in rt.read_json(run/'prior_model_hashes.json').items():
            if file_sha256(Path(path))!=digest:raise RuntimeError('Prior release changed')
    return check_leakage(pd.read_csv(run/'split_manifest.csv'))


def prepare_sequences(run):
    run=Path(run);verify_run(run)
    if (run/'sequences/checksums.json').exists():rt.verify_seal(run/'sequences');return dict(status='already_prepared')
    options=rt.read_json(run/'run_config.json')['options'];catalog=pd.read_csv(run/'split_manifest.csv')
    for length in [16,32]:
        contract=sequence_contract(length,**options['sequence']);records=[];resets=[]
        for row in catalog.to_dict('records'):
            x,eligible,ticks,tracks,sessions=load_clip(run,row)
            windows,boundaries=window_records(ticks,eligible,tracks,sessions,contract,1)
            for w in windows:
                w.update(clip_id=row['clip_id'],source_group_id=row['source_group_id'],person_id=row['person_id'],
                         session_id=row['session_id'],camera_id=row['camera_id'],label=int(row['label']),partition=row['partition'],
                         length=length,training_selected=(w['target_frame']-(length-1))%contract['training_stride']==0)
                if row['partition']=='excluded':w['accepted']=False;w['rejection_reason']='excluded_source_take'
                records.append(w)
            resets.extend(dict(clip_id=row['clip_id'],frame_index=i,reasons=';'.join(reason)) for i,reason in boundaries.items())
        table=pd.DataFrame(records)
        table.to_csv(run/f'sequences/windows{length}.csv.gz',index=False,compression='gzip')
        pd.DataFrame(resets).to_csv(run/f'sequences/resets{length}.csv',index=False)
        rt.write_json(run/f'sequences/contract{length}.json',contract,exclusive=True)
        table.groupby(['partition','accepted','rejection_reason'],dropna=False).size().reset_index(name='windows').to_csv(run/f'sequences/rejections{length}.csv',index=False)
    # Freeze common target coverage before any model sees validation scores.
    a=pd.read_csv(run/'sequences/windows16.csv.gz');b=pd.read_csv(run/'sequences/windows32.csv.gz')
    keys=['clip_id','target_frame','partition']
    common=a[a.accepted][keys].merge(b[b.accepted][keys],on=keys,validate='one_to_one')
    common.to_csv(run/'sequences/common_targets.csv',index=False)
    for part in PARTS:
        ids=set(common.loc[common.partition==part,'clip_id'])
        if set(catalog.loc[catalog.clip_id.isin(ids),'label'])!=set(range(6)):raise ValueError('Common GRU/MLP target coverage lacks a class')
    rt.seal(run/'sequences')
    return dict(common_targets=common.groupby('partition').size().to_dict(),padding='none',test_model_predictions=False)


class TakeData(rt.TrainingData):
    def __init__(self,run,kind,partitions,device,scope='matched',full=False):
        run=Path(run);verify_run(run)
        rt.verify_seal(run/'sequences')
        self.run=run;self.device=device;self.input_dim=52;self.scope=scope;self.kind=kind
        catalog=pd.read_csv(run/'split_manifest.csv')
        if full:catalog=catalog[catalog.usable_frames>0].copy();catalog['partition']='train'
        else:catalog=catalog[catalog.partition.isin(partitions)]
        length=int(kind.replace('gru','')) if kind.startswith('gru') else 1
        windows=pd.read_csv(run/f'sequences/windows{length}.csv.gz') if length>1 else None
        common=pd.read_csv(run/'sequences/common_targets.csv')
        common_keys=set(common.clip_id+':'+common.target_frame.astype(str))
        checks=rt.read_json(run/'processed/checksums.json')
        xs=[];ys=[];self.clips=[];self.metadata=[];self.target_keys=[];offset=0
        for row in catalog.to_dict('records'):
            file=run/'processed'/f'{row["clip_id"]}.npz'
            if file_sha256(file)!=checks[file.name]:raise RuntimeError('Immutable feature mismatch')
            with np.load(file,allow_pickle=False) as z:
                x=z['x'].reshape(-1,52);eligible=z['eligible'];ticks=z['capture_ns']
            part=row['partition']
            if length==1:
                targets=np.flatnonzero(eligible);inputs=x[targets]
            else:
                piece=windows[(windows.clip_id==row['clip_id']) & windows.accepted]
                if part=='train':piece=piece[piece.training_selected]
                targets=piece.target_frame.to_numpy(np.int64)
                indices=targets[:,None]+np.arange(-length+1,1,dtype=np.int64)[None,:]
                inputs=x[indices]
            if not eligible[targets].all():raise RuntimeError('Ineligible target entered training/evaluation')
            xs.append(inputs);ys.append(np.full(len(targets),int(row['label']),np.int64))
            for target in targets:
                key=f'{row["clip_id"]}:{target}'
                self.target_keys.append(key)
                self.metadata.append(dict(target_key=key,clip_id=row['clip_id'],target_frame=int(target),capture_ns=int(ticks[target]),
                        source_group_id=row['source_group_id'],camera_id=row['camera_id'],person_id=row['person_id'],partition=part))
            self.clips.append(dict(start=offset,end=offset+len(targets),label=int(row['label']),clip_id=row['clip_id'],
                    person=row['person_id'],camera=row['camera_id'],camera_view=row['camera_view'],source_group_id=row['source_group_id'],split=part))
            offset+=len(targets)
        if not xs or offset==0:raise ValueError('No training/evaluation targets')
        self.x=np.concatenate(xs).astype(np.float32);self.y=np.concatenate(ys)
        self.valid=np.ones(offset,bool);self.common=np.array([key in common_keys for key in self.target_keys])
        self.target_keys=np.asarray(self.target_keys)
        if not np.isfinite(self.x).all() or self.x.shape[-1]!=52:raise ValueError('Invalid input features')
        if length>1 and self.x.shape[1:]!=(length,52):raise ValueError('Invalid GRU input shape')
        self.xt=torch.as_tensor(self.x,device=device);self.yt=torch.as_tensor(self.y,device=device)

    def ids(self,clips):
        ids=super().ids(clips)
        if self.scope=='matched' and clips and all(c['split']!='train' for c in clips):ids=ids[self.common[ids]]
        return ids


def candidate_config(run,kind,seed):
    cap=rt.read_json(Path(run)/'run_config.json')['options']['per_source_window_cap']
    return {**rt.DEFAULT,'model':'mlp_wide' if kind=='mlp_control' else kind,'seed':int(seed),
            'epochs':100 if kind=='mlp_control' else 80,'batch_size':128 if kind=='mlp_control' else 64,
            'patience':12,'gradient_clip':0. if kind=='mlp_control' else 1.,'quota':0 if kind=='mlp_control' else cap,
            'selection_metric':'frame_macro_f1','sequence_length':1 if kind=='mlp_control' else int(kind[3:]),
            'hidden_dim':64,'layers':1,'bidirectional':False,'evaluation_scope':'same frozen GRU16/32 target intersection'}


def candidate_dir(run,kind,seed):return Path(run)/kind/f'seed{seed}'


def evaluate_lineage(model,data,clips):
    score,frames=rt.evaluate(model,data,clips)
    ids=data.ids(clips)
    for key in ['target_key','target_frame','capture_ns']:
        frames[key]=[data.metadata[i][key] for i in ids]
    # Switching is counted only on adjacent original frames of one camera clip.
    switches=0;transitions=0
    for _,piece in frames.groupby('clip_id',sort=False):
        piece=piece.sort_values('target_frame')
        adjacent=np.diff(piece.target_frame.to_numpy())==1
        switches+=int(((np.diff(piece.pred.to_numpy())!=0)&adjacent).sum());transitions+=int(adjacent.sum())
    score['switch_rate']=switches/max(1,transitions);score['switches']=switches;score['adjacent_transitions']=transitions
    score['low_confidence_fraction']=float((frames.confidence<.7).mean())
    score['target_coverage_of_model_available']=len(ids)/max(1,sum(c['end']-c['start'] for c in clips))
    return score,frames


def load_candidate(run,kind,seed,device):
    folder=candidate_dir(run,kind,seed)
    saved=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    model=build_classifier_model(saved['model_name'],saved['model_config']['dropout']).to(device)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model.eval(),rt.read_json(folder/'result.json')


def train_model(run,kind,device,resume=False):
    run=Path(run);verify_run(run)
    data=TakeData(run,kind,['train','val'],device)
    config=candidate_config(run,kind,302)
    smoke={**config,'epochs':3,'patience':3}
    rt.run_candidate(run,smoke,device,f'smoke_{kind}',resume,data=data,experiment_dir=run/kind/'smoke')
    history=pd.read_csv(run/kind/'smoke/history.csv')
    if history.train_loss.iloc[-1]>=history.train_loss.iloc[0]:raise RuntimeError('Real-data CUDA smoke loss did not decrease')
    results=[]
    for seed in [302,303,304]:
        config=candidate_config(run,kind,seed);folder=candidate_dir(run,kind,seed)
        result=rt.run_candidate(run,config,device,f'{kind}_seed{seed}',resume,data=data,experiment_dir=folder)
        model,_=load_candidate(run,kind,seed,device)
        val=[c for c in data.clips if c['split']=='val']
        matched,pred=evaluate_lineage(model,data,val)
        rt.save_metrics(matched,pred,folder/'validation_matched')
        data.scope='own';own,own_pred=evaluate_lineage(model,data,val);data.scope='matched'
        rt.save_metrics(own,own_pred,folder/'validation_own_coverage')
        torch.cuda.synchronize() if str(device).startswith('cuda') else None
        with torch.inference_mode():
            sample=data.xt[data.ids(val)[:1]]
            for _ in range(20):model(sample)
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(200):model(sample)
            if str(device).startswith('cuda'):torch.cuda.synchronize()
        measured=dict(parameters=sum(p.numel() for p in model.parameters()),batch_one_ms=(time.perf_counter()-start)*1000/200,
                      latency_scope='Warmed classifier only, batch one, synchronized CUDA wall time, excludes YOLO.',
                      own_prediction_frames=own['frames'],matched_prediction_frames=matched['frames'])
        rt.write_json(folder/'latency_coverage.json',measured)
        results.append(dict(kind=kind,seed=seed,best_epoch=result['best_epoch'],epochs=result['epochs_completed'],
                            **{k:v for k,v in matched.items() if isinstance(v,(int,float))},**measured))
    pd.DataFrame(results).to_csv(run/kind/'validation_comparison.csv',index=False)
    return results


def compare(run,device):
    run=Path(run);records=[]
    for kind in MODELS:
        records.extend(pd.read_csv(run/kind/'validation_comparison.csv').to_dict('records'))
    table=pd.DataFrame(records);(run/'comparisons').mkdir(exist_ok=True)
    table.to_csv(run/'comparisons/validation_runs.csv',index=False)
    ranking=table.groupby('kind').agg(mean_frame_f1=('frame_macro_f1','mean'),std_frame_f1=('frame_macro_f1','std'),
                mean_subject_f1_fixed6=('subject_macro_f1','mean'),mean_take_f1=('source_macro_f1','mean'),
                mean_clip_f1=('clip_macro_f1','mean'),mean_switch_rate=('switch_rate','mean'),median_epoch=('best_epoch','median'),
                mean_latency_ms=('batch_one_ms','mean'),parameters=('parameters','first')).reset_index()
    ranking=ranking.sort_values(['mean_frame_f1','mean_take_f1'],ascending=False)
    ranking.to_csv(run/'comparisons/model_comparison.csv',index=False)
    indexed=ranking.set_index('kind');base=indexed.loc['mlp_control']
    challengers=indexed.drop('mlp_control').sort_values('mean_frame_f1',ascending=False)
    decisions=[]
    for kind,row in challengers.iterrows():
        gain=float(row.mean_frame_f1-base.mean_frame_f1)
        stability=bool(base.mean_switch_rate>.005 and row.mean_switch_rate<=base.mean_switch_rate*.75 and gain>=-.005)
        # Same targets for all models, three seeds; no coverage-based cherry-pick.
        convincing=bool(gain>=.01 or stability)
        per_class=[]
        for seed in [302,303,304]:
            cm=rt.read_json(candidate_dir(run,kind,seed)/'validation_matched/metrics.json')
            bm=rt.read_json(candidate_dir(run,'mlp_control',seed)/'validation_matched/metrics.json')
            per_class.append([cm['classification_report'][str(i)]['recall']-bm['classification_report'][str(i)]['recall'] for i in range(6)])
        min_recall_delta=float(np.mean(per_class,axis=0).min())
        convincing&=min_recall_delta>=-.03
        decisions.append(dict(kind=kind,frame_f1_gain=gain,stability_gate=stability,min_class_recall_delta=min_recall_delta,promote=bool(convincing)))
    approved=[d for d in decisions if d['promote']]
    winner=max(approved,key=lambda d:d['frame_f1_gain'])['kind'] if approved else 'mlp_control'
    # Source-take-aware paired uncertainty: average each take over seeds, then
    # resample takes, never individual correlated frames.
    take_rows=[]
    for kind in MODELS:
        for seed in [302,303,304]:
            pred=pd.read_csv(candidate_dir(run,kind,seed)/'validation_matched/frame_predictions.csv.gz')
            grouped=pred.assign(correct=pred.label.eq(pred.pred)).groupby('source_group_id').correct.mean()
            take_rows.extend(dict(kind=kind,seed=seed,take=take,accuracy=float(value)) for take,value in grouped.items())
    take_table=pd.DataFrame(take_rows);take_table.to_csv(run/'comparisons/paired_take_accuracy.csv',index=False)
    means=take_table.groupby(['take','kind']).accuracy.mean().unstack();rng=np.random.default_rng(302)
    intervals={}
    for kind in ['gru16','gru32']:
        delta=(means[kind]-means.mlp_control).to_numpy()
        samples=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
        intervals[kind]=dict(mean_take_accuracy_gain=float(delta.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),unit='original source take, camera views and seeds averaged')
    selection=dict(winner=winner,representative_seed=302,internal_test_label=TEST_LABEL,
                   fixed_final_epochs=max(1,int(round(indexed.loc[winner,'median_epoch']))),
                   decisions=decisions,take_bootstrap=intervals,internal_test_accessed=False,
                   promotion_rule='>=0.01 matched mean frame Macro-F1 gain OR >=25% flicker reduction with <=0.005 F1 loss, and no class recall loss >0.03.',
                   feature_contract=FEATURE_VERSION,gru_hidden=64,seeds=[302,303,304],coverage='same frozen target intersection for every candidate')
    path=run/'selection.json'
    if path.exists() and rt.read_json(path)!=selection:raise RuntimeError('Selection already frozen')
    if not path.exists():rt.write_json(path,selection,exclusive=True);path.chmod(0o444)
    return selection


def internal_test(run,device):
    run=Path(run);output=run/'internal_test/metrics.json'
    if output.exists():return rt.read_json(output)
    selection=rt.read_json(run/'selection.json');kind=selection['winner'];seed=selection['representative_seed']
    folder=run/'internal_test';folder.mkdir(exist_ok=True)
    rt.write_json(folder/'evaluation_started.json',dict(selection_sha256=file_sha256(run/'selection.json'),
                    checkpoint_sha256=file_sha256(candidate_dir(run,kind,seed)/'best.pt'),label=TEST_LABEL),exclusive=True)
    model,_=load_candidate(run,kind,seed,device)
    # One forward pass on winner's own available test targets. Common-scope
    # metrics are derived from those cached probabilities/predictions if needed.
    data=TakeData(run,kind,['test'],device,scope='own')
    score,pred=evaluate_lineage(model,data,data.clips)
    score['internal_test_loss']=score.pop('validation_loss')
    score.update(evaluation_label=TEST_LABEL,winner=kind,seed=seed,test_evaluation_count=1,
                 fresh_take_split_checkpoint=True,old_full_data_model_scored=False)
    rt.save_metrics(score,pred,folder)
    return score


def install_checkpoint(run,kind,checkpoint,folder,scope,reserved=False):
    run=Path(run);folder=Path(folder)
    if reserved:
        if not folder.is_dir() or any(folder.iterdir()):raise ValueError('Reserved new model directory must be empty')
    else:folder.mkdir(parents=True,exist_ok=False)
    contract=rt.read_json(run/'feature_contract.json')
    source=Path(rt.read_json(run/'run_config.json')['source_run'])
    labels=rt.read_json(source/'snapshot/config/labels.json')
    names=[r['name'] for r in sorted(labels['labels'],key=lambda r:r['id'])]
    checkpoint.update(stage='final',feature_contract=contract,label_names=names,
                      training_scope=scope,external_evaluation_pending=True)
    rt.atomic_torch_save(checkpoint,folder/'model_final.pt')
    rt.write_json(folder/'model_config.json',checkpoint['model_config'],exclusive=True)
    rt.write_json(folder/'feature_contract.json',contract,exclusive=True)
    rt.write_json(folder/'label_map.json',labels,exclusive=True)
    rt.write_json(folder/'split_manifest.json',rt.read_json(run/'split_manifest.json'),exclusive=True)
    rt.write_json(folder/'training_manifest.json',dict(scope=scope,split_sha256=file_sha256(run/'split_manifest.json'),
                  feature_hashes_sha256=file_sha256(run/'processed/checksums.json'),
                  data_hashes=rt.read_json(run/'audit/source_inventory.json'),internal_test_scored=False,
                  external_evaluation_pending=True),exclusive=True)
    if kind.startswith('gru'):
        rt.write_json(folder/'sequence_contract.json',rt.read_json(run/f'sequences/contract{kind[3:]}.json'),exclusive=True)
    return folder


def finalize(run,device,resume=False):
    run=Path(run).resolve();marker=run/'final_model.json'
    if marker.exists():return rt.read_json(marker)
    selection=rt.read_json(run/'selection.json');winner=selection['winner']
    raw=Path(rt.read_json(run/'run_config.json')['raw_root'])
    if winner=='mlp_control':
        # Same architecture/features/hyperparameters as the existing all-data
        # release. Seen-subject take validation does not justify replacing its
        # historical subject-development schedule just to fit longer.
        folder=raw.parents[1]/'models/mlp_wide_full_real_dataset/v001'
        summary=rt.read_json(folder/'full_data_training_manifest.json')
        current=pd.read_csv(run/'split_manifest.csv')
        assert summary['eligible_frames']==int(current.usable_frames.sum())
        assert set(summary['subjects'])==set(current.person_id)
        value=dict(path=str(folder),winner=winner,reused_existing_full_data=True,eligible_frames=summary['eligible_frames'],
                   epochs=rt.read_json(folder/'training_summary.json')['epochs'],
                   reason='GRUs failed matched validation. Preserve the existing all-data Wide candidate and historical subject-development schedule; no unnecessary refit.',
                   take_control_median_best_epoch=selection['fixed_final_epochs'],external_evaluation_pending=True)
    else:
        cfg=candidate_config(run,winner,10302)
        data=TakeData(run,winner,['train'],device,scope='own',full=True)
        out=run/winner/'full_fit'
        rt.run_candidate(run,cfg,device,'full_gru_fit',resume,final_epochs=selection['fixed_final_epochs'],data=data,experiment_dir=out)
        saved=torch.load(out/'trained.pt',map_location='cpu',weights_only=True)
        parent=raw.parents[1]/'models/gru_full_real_dataset';parent.mkdir(parents=True,exist_ok=True)
        # Reserve a model version without overwriting a previous release.
        reserved=rt.version_dir(parent)
        folder=install_checkpoint(run,winner,saved,reserved,'all usable real windows from all 13 subjects; no held-out final score',reserved=True)
        shutil.copyfile(out/'history.csv',folder/'training_history.csv')
        value=dict(path=str(folder),winner=winner,reused_existing_full_data=False,training_windows=len(data.x),
                   eligible_source_frames=int(pd.read_csv(run/'split_manifest.csv').usable_frames.sum()),epochs=selection['fixed_final_epochs'],external_evaluation_pending=True)
    rt.write_json(marker,value,exclusive=True)
    # Both real trained GRUs remain loadable/exportable research artifacts even
    # when the validation-selected architecture remains MLP.
    for kind in ['gru16','gru32']:
        folder=run/kind/'deployment'
        if not folder.exists():
            saved=torch.load(candidate_dir(run,kind,302)/'best.pt',map_location='cpu',weights_only=True)
            install_checkpoint(run,kind,saved,folder,'RESEARCH: take-level training partition only; not promoted over MLP')
            shutil.copyfile(candidate_dir(run,kind,302)/'history.csv',folder/'training_history.csv')
    return value


def deploy(run):
    import subprocess
    import sys
    from scripts.export_onnx import export
    from inference.pipeline import PostureClassifier
    from shared.sequence_data import TemporalSequenceBuffer
    run=Path(run).resolve();final=rt.read_json(run/'final_model.json')
    targets=[('selected_full_model',Path(final['path']),'mlp_control' if final['winner']=='mlp_control' else final['winner']),
             ('research_gru16',run/'gru16/deployment','gru16'),('research_gru32',run/'gru32/deployment','gru32')]
    catalog=pd.read_csv(run/'split_manifest.csv');windows=pd.read_csv(run/'sequences/windows32.csv.gz')
    candidates=windows[windows.partition.eq('train')&windows.accepted].groupby('clip_id').size()
    smoke_row=catalog[catalog.clip_id.isin(candidates[candidates>=16].index)].iloc[0].to_dict()
    raw=Path(rt.read_json(run/'run_config.json')['raw_root']);root=raw.parents[2]
    (run/'inference_smoke').mkdir(exist_ok=True);reports=[]
    for tag,folder,kind in targets:
        if not (folder/'model.onnx').exists():export(folder)
        data=TakeData(run,kind,['train','val'],'cpu',scope='own')
        pt=PostureClassifier(folder,'pt');onnx=PostureClassifier(folder,'onnx')
        difference=0.
        with torch.inference_mode():
            for start in range(0,len(data.x),128):
                x=data.x[start:start+128]
                a=pt.model(torch.from_numpy(x)).softmax(1).numpy()
                b=torch.from_numpy(onnx.session.run(['logits'],{'features':x})[0]).softmax(1).numpy()
                if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite deployment output')
                difference=max(difference,float(np.abs(a-b).max()))
        if difference>1e-5:raise RuntimeError('Real GRU/MLP ONNX probability parity failed')
        for x in data.x[:16]:np.testing.assert_allclose(pt.predict(x),onnx.predict(x),atol=1e-5,rtol=1e-5)
        times=[]
        for x in data.x[:50]:onnx.predict(x)
        for i in range(1000):
            start=time.perf_counter_ns();p=onnx.predict(data.x[i%len(data.x)]);times.append((time.perf_counter_ns()-start)/1e6)
            assert p.shape==(6,) and np.isclose(p.sum(),1)
        marker=run/f'inference_smoke/{tag}.json'
        if not marker.exists():
            command=[sys.executable,str(root/'inference/run.py'),'--source',str(rt.inside(raw,smoke_row['video_path'])),
                    '--mode','video','--model',str(folder),'--pose',str(raw.parents[1]/'pretrained/yolo11n-pose.pt'),
                    '--device','cuda:0','--backend','onnx','--cls-device','cpu','--headless','--name','take_gru_'+tag]
            result=subprocess.run(command,cwd=root,text=True,encoding='utf-8',capture_output=True)
            (run/f'inference_smoke/{tag}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('End-to-end development inference failed; see smoke log')
            output=max((raw.parents[1]/'inference').glob(f'take_gru_{tag}_v*/performance.json'),key=lambda p:p.stat().st_mtime_ns)
            rt.write_json(marker,dict(command=command,source_partition='train',clip_id=smoke_row['clip_id'],output=str(output.parent),
                                     performance=rt.read_json(output),external_test=False),exclusive=True)
        reports.append(dict(tag=tag,model=str(folder),probability_max_abs_error=difference,parity_sequences_or_frames=len(data.x),
                            sequence_length=pt.sequence_length,finite_six_class_probabilities=True,labels_match=pt.labels==onnx.labels,
                            onnx_cpu_median_ms=float(np.median(times)),onnx_cpu_p95_ms=float(np.quantile(times,.95)),
                            smoke=rt.read_json(marker)))
    # Exact timing/buffer coverage replay on cached real training/validation
    # streams; this uses no model or labels from the internal-test partition.
    replay_counts={}
    for length in [16,32]:
        checked=0
        for row in catalog[catalog.partition.isin(['train','val'])].to_dict('records'):
            x,eligible,ticks,tracks,sessions=load_clip(run,row)
            contract=rt.read_json(run/f'sequences/contract{length}.json');buffer=TemporalSequenceBuffer(contract)
            expected,_=window_records(ticks,eligible,tracks,sessions,contract)
            actual=[]
            for i in range(len(x)):
                seq=buffer.push(x[i],ticks[i],eligible[i],tracks[i])
                if seq is not None:
                    actual.append(i);np.testing.assert_array_equal(seq,x[i-length+1:i+1])
            if actual!=[r['target_frame'] for r in expected if r['accepted']]:raise RuntimeError('Offline/online causal buffer coverage mismatch')
            checked+=len(x)
        replay_counts[str(length)]=checked
    verify_run(run,sources=True)
    value=dict(verified=True,models=reports,real_buffer_replay_frames=replay_counts,
                source_and_previous_release_hashes_unchanged=True,external_accuracy_test_executed=False,live_camera_accessed=False)
    rt.write_json(run/'deployment_verification.json',value)
    for tag,folder,kind in targets:
        # Do not write any new reports inside preserved older model releases.
        if folder.is_relative_to(run):rt.write_json(folder/'deployment_verification.json',next(r for r in reports if r['tag']==tag))
    return value


def md_table(df,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in df[columns].to_dict('records'):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row.values())+' |')
    return '\n'.join(lines)


def report(run):
    from sklearn.metrics import f1_score,accuracy_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from zipfile import ZipFile,ZIP_DEFLATED
    run=Path(run).resolve();raw=Path(rt.read_json(run/'run_config.json')['raw_root']);root=raw.parents[2]
    selection=rt.read_json(run/'selection.json');internal=rt.read_json(run/'internal_test/metrics.json')
    final=rt.read_json(run/'final_model.json');deployment=rt.read_json(run/'deployment_verification.json')
    audit_result=rt.read_json(run/'audit/training_preflight.json');quality=rt.read_json(run/'split_quality.json')
    catalog=pd.read_csv(run/'split_manifest.csv');ranking=pd.read_csv(run/'comparisons/model_comparison.csv')
    runs=pd.read_csv(run/'comparisons/validation_runs.csv');common=pd.read_csv(run/'sequences/common_targets.csv')
    frames_total={part:int(catalog.loc[catalog.partition==part,'usable_frames'].sum()) for part in PARTS}
    split_rows=[];label_rows=[]
    for part,summary in quality['counts'].items():
        split_rows.append(dict(partition=part,takes=summary['takes'],camera_clips=summary['camera_clips'],subjects=summary['subjects'],
                               eligible_frames=summary['eligible_frames'],front_frames=summary['camera_eligible_frames'].get('cam01',0),
                               oblique_frames=summary['camera_eligible_frames'].get('cam02',0)))
        for label in range(6):label_rows.append(dict(partition=part,label=label,takes=summary['label_takes'].get(str(label),0)))
    split_table=pd.DataFrame(split_rows);label_table=pd.DataFrame(label_rows).pivot(index='label',columns='partition',values='takes').reset_index()
    coverage=[];sequence_rows=[];subject_rows=[];majority=[];errors=[];camera_rows=[]
    for length in [16,32]:
        windows=pd.read_csv(run/f'sequences/windows{length}.csv.gz')
        for part in PARTS:
            w=windows[windows.partition==part];accepted=w[w.accepted]
            sequence_rows.append(dict(length=length,partition=part,candidates=len(w),accepted=len(accepted),
                          source_takes=int(accepted.source_group_id.nunique()),camera_clips=int(accepted.clip_id.nunique()),
                          median_window_span_s=float(((accepted.end_capture_ns-accepted.start_capture_ns)/1e9).median()),
                          rejected=len(w)-len(accepted),training_stride_selected=int(accepted.training_selected.sum())))
        # Complete source-index lineage supplement; membership stays frozen.
        lineage=windows.merge(catalog[['clip_id','source_frame_start']],on='clip_id',validate='many_to_one')
        lineage['source_frame_end_exclusive']=lineage.source_frame_start+lineage.end_frame_exclusive
        lineage['target_source_frame']=lineage.source_frame_start+lineage.target_frame
        lineage['source_frame_start']=lineage.source_frame_start+lineage.start_frame
        lineage.to_csv(run/f'audit/window_source_lineage{length}.csv.gz',index=False,compression='gzip')
    for kind in MODELS:
        for seed in [302,303,304]:
            folder=candidate_dir(run,kind,seed)
            for scope in ['validation_matched','validation_own_coverage']:
                metric=rt.read_json(folder/scope/'metrics.json')
                coverage.append(dict(kind=kind,seed=seed,scope=scope,predicted_frames=metric['frames'],
                         eligible_target_coverage=metric['frames']/frames_total['val'],
                         raw_frame_coverage=metric['frames']/int(catalog.loc[catalog.partition=='val','frames'].sum()),
                         clip_coverage=metric['clips'],take_coverage=metric['source_takes'],frame_f1=metric['frame_macro_f1']))
            pred=pd.read_csv(folder/'validation_matched/frame_predictions.csv.gz')
            # Compare exact target identities, not merely sample counts.
            expected=set(common.loc[common.partition=='val','clip_id']+':'+common.loc[common.partition=='val','target_frame'].astype(str))
            if set(pred.target_key)!=expected:raise RuntimeError('Candidate comparison target coverage differs')
            for person,p in pred.groupby('person'):
                subject_rows.append(dict(kind=kind,seed=seed,person=person,frames=len(p),present_labels=','.join(map(str,sorted(p.label.unique()))),
                    present_label_macro_f1=float(f1_score(p.label,p.pred,labels=sorted(p.label.unique()),average='macro',zero_division=0))))
            for unit in ['clip_id','source_group_id']:
                votes=pred.groupby(unit).agg(label=('label','first'),pred=('pred',lambda p:int(p.value_counts().index[0])))
                majority.append(dict(kind=kind,seed=seed,unit=unit,majority_vote_accuracy=float(accuracy_score(votes.label,votes.pred))))
            error=pred[pred.label!=pred.pred].groupby(['person','camera','label','pred']).size().reset_index(name='frames')
            error['kind']=kind;error['seed']=seed;errors.extend(error.to_dict('records'))
            camera_rows.extend(dict(kind=kind,seed=seed,**c) for c in rt.read_json(folder/'validation_matched/metrics.json')['per_camera'])
    pd.DataFrame(coverage).to_csv(run/'comparisons/coverage.csv',index=False)
    pd.DataFrame(sequence_rows).to_csv(run/'sequences/sequence_summary.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(run/'comparisons/subject_scores_present_classes.csv',index=False)
    pd.DataFrame(majority).to_csv(run/'comparisons/majority_vote_accuracy.csv',index=False)
    pd.DataFrame(errors).to_csv(run/'comparisons/error_pairs.csv',index=False)
    pd.DataFrame(camera_rows).to_csv(run/'comparisons/camera_comparison.csv',index=False)
    # Additional internal-test summaries use cached predictions only, with no
    # second model forward pass and no change to the frozen selection.
    pred=pd.read_csv(run/'internal_test/frame_predictions.csv.gz')
    test_keys=set(common.loc[common.partition=='test','clip_id']+':'+common.loc[common.partition=='test','target_frame'].astype(str))
    matched=pred[pred.target_key.isin(test_keys)]
    rt.write_json(run/'internal_test/common_target_metrics_from_cached_predictions.json',dict(frames=len(matched),
                    frame_macro_f1=float(f1_score(matched.label,matched.pred,labels=list(range(6)),average='macro',zero_division=0)),
                    frame_accuracy=float(accuracy_score(matched.label,matched.pred)),derived_from_cached_once_only_predictions=True))
    per_subject=[]
    for person,p in pred.groupby('person'):
        per_subject.append(dict(person=person,frames=len(p),present_labels=','.join(map(str,sorted(p.label.unique()))),
                        present_label_macro_f1=float(f1_score(p.label,p.pred,labels=sorted(p.label.unique()),average='macro',zero_division=0))))
    pd.DataFrame(per_subject).to_csv(run/'internal_test/per_subject_present_class_metrics.csv',index=False)
    # Classifier/coverage/flicker comparison uses common frozen validation data.
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    axes[0].bar(ranking.kind,ranking.mean_frame_f1,yerr=ranking.std_frame_f1,capsize=4)
    axes[0].set(ylabel='Frame Macro-F1',ylim=(0,1),title='Matched validation, 3 seeds')
    own=pd.DataFrame(coverage);own=own[own.scope=='validation_own_coverage'].groupby('kind').eligible_target_coverage.mean().reindex(ranking.kind)
    axes[1].bar(own.index,own);axes[1].set(ylim=(0,1),ylabel='Fraction of eligible frames',title='Own validation prediction coverage')
    axes[2].bar(ranking.kind,ranking.mean_switch_rate*100);axes[2].set(ylabel='Switches / adjacent predictions (%)',title='Matched raw prediction switching')
    for ax in axes:ax.tick_params(axis='x',rotation=20)
    fig.savefig(run/'comparisons/validation_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for ax,kind in zip(axes,MODELS):
        history=pd.read_csv(candidate_dir(run,kind,302)/'history.csv')
        ax.plot(history.epoch,history.val_frame_f1,label='matched validation frame F1')
        ax.set(title=kind,xlabel='Epoch',ylabel='Frame Macro-F1',ylim=(0,1));ax.legend(fontsize=7)
    fig.savefig(run/'comparisons/learning_curves.png',dpi=150);plt.close(fig)
    cpu=pd.DataFrame([dict(model=r['tag'],median_ms=r['onnx_cpu_median_ms'],p95_ms=r['onnx_cpu_p95_ms'],
                           max_probability_error=r['probability_max_abs_error'],smoke_frames=r['smoke']['performance']['frames'],
                           smoke_predictions=r['smoke']['performance']['predicted_frames'],smoke_fps=r['smoke']['performance']['end_to_end_fps']) for r in deployment['models']])
    cpu.to_csv(run/'comparisons/onnx_latency_smoke.csv',index=False)
    verify_run(run,sources=True)
    full_tests=(run/'regression_full.log').read_text(encoding='utf-8')
    focused_path=run/'regression_take_gru_final.log'
    focused_tests=(focused_path if focused_path.exists() else run/'regression_take_gru.log').read_text(encoding='utf-8')
    full_summary='\n'.join(s for s in full_tests.splitlines() if 'passed,' in s or s.startswith('FAILED '))
    focused_summary='\n'.join(s for s in focused_tests.splitlines() if 'passed,' in s)
    coverage_table=pd.DataFrame(coverage);coverage_table=coverage_table[(coverage_table.seed==302)&coverage_table.scope.eq('validation_own_coverage')]
    source=Path(rt.read_json(run/'run_config.json')['source_run'])
    names=['shared/take_gru_training.py','shared/sequence_data.py','shared/gru_model.py','shared/mlp_model.py','shared/real_training.py',
           'inference/pipeline.py','inference/run.py','scripts/export_onnx.py','scripts/train_poseforge.py','scripts/evaluate_real_world.py']
    code=run/'source_code';code.mkdir(exist_ok=True)
    for name in names:
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    rt.write_json(run/'code_hashes.json',{name:file_sha256(root/name) for name in names})
    rt.write_json(run/'environment.json',dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                  feature_contract=FEATURE_VERSION,input_dim=52,all_training_device='cuda:0'),exclusive=True)
    text=f'''# PoseForge take-level MLP/GRU training report

## Outcome

Actual CUDA training completed for a fresh MLP Wide, GRU-16 and GRU-32, each at
seeds 302/303/304. **MLP Wide won matched validation. Neither GRU is promoted.**
The existing all-data MLP remains the deployment candidate: `{final['path']}`.
Both genuinely trained GRUs are saved as exportable research models and their
causal real-time inference paths passed PT/ONNX and development-video checks.
The internal take-level test ran once for the frozen fresh MLP winner. No
external accuracy evaluation or live-camera access occurred, per user instruction.

## Audited data and frozen take split

Current real data: {audit_result['subject_count']} subjects, {audit_result['source_takes']} original
takes, {audit_result['clips']} camera clips, {audit_result['frames']:,} raw frames,
{audit_result['eligible_frames']:,} eligible observations. All 30 recovered files
remain recognized; 16 camera files remain pose-empty. Existing confidence/missing
masks and quality exclusions were retained. No labels or original files changed.

Allocation seed 302 ranked 512 label-stratified candidate allocations by predefined
train-subject coverage, aggregate six-class/GRU32 feasibility, subject diversity,
and eligible-label frame balance. Candidate 134 was frozen before any training.
No model score selected the split. Both cameras and every derived observation of
an original source take stay together. Track IDs never identify dataset people.
No within-video temporal split or embargo was used.

{md_table(split_table,['partition','takes','camera_clips','subjects','eligible_frames','front_frames','oblique_frames'])}

Original takes by class:

{md_table(label_table,['label','train','val','test','excluded'])}

Only one group is entirely excluded: TDUY/session_20261027_1124/label_5/take_001,
because both views have zero eligible observations. Partial-pose camera clips in
other groups remain represented; unusable target frames/windows are excluded by
the quality/timing rules. Validation has 12 subjects, internal test 9, all seen in
the 13-subject training pool. Per-person label coverage across all splits is not
required: most person-label pairs have only one take. All six labels occur at
aggregate partition level. `split_manifest.*`, `split_quality.json`, audit CSVs,
and `leakage_checks.json` preserve exact source identities, hashes and coverage.

## Real sequence extraction

Inputs are `[batch,16,52]` or `[batch,32,52]`, using unchanged
`{FEATURE_VERSION}` features. Each window belongs to one person/session/take/camera
and one partition. Original timestamp/frame order is retained. Confidence/masks
remain measured; no coordinates or transition labels are fabricated.

Training stride is four; evaluation stride is one. Each original take/camera is
capped at 48 training windows per epoch. A target must be eligible, with >=80%
usable observations in its context. Gaps >0.25 seconds, backward timestamps,
selected-track/session changes, and >3 consecutive missing observations reset or
reject spanning windows. Repeated ticks are retained and counted in lineage.
Missing frames remain in the sequence as their original masked features; they are
not compressed out. No padding is used. Warmup, missing-target, low-coverage and
reset-crossing rejections are recorded for every candidate window.

{md_table(pd.DataFrame(sequence_rows),['length','partition','accepted','rejected','source_takes','camera_clips','median_window_span_s'])}

`sequences/windows16.csv.gz` and `windows32.csv.gz` record local frame bounds,
target frame, capture-time endpoints, source identity, partition and reasons.
Immutable per-clip NPZs retain full timestamp vectors. The audit's window-source
lineage supplements map those bounds to original source-frame indices. Timing
features were not appended: this controlled baseline remains 52-dimensional.
Mostly static/held posture takes provide limited motion diversity; repeated,
overlapping windows do not create independent recordings.

## Actual training and matched validation

GPU: NVIDIA GeForce RTX 2050; PyTorch {torch.__version__}, CUDA {torch.version.cuda}.
AdamW lr 0.001, weight_decay 0.001, dropout 0.15, ordinary six-class CE, no weighting,
smoothing or augmentation. MLP: 52–256–128–6, batch 128, max 100 epochs.
Each GRU: unidirectional single layer, hidden 64, zero initial hidden state per
window, classification-head dropout 0.15, batch 64, max 80 epochs, gradient clipping
1.0. Single-layer GRU built-in dropout is not used. Early stopping patience: 12.
All twelve CUDA runs completed: three decreasing-loss real-data smoke runs plus
nine full candidates. Every candidate has best/resume checkpoints, config, history,
curves, per-class/camera/subject metrics, confusion matrices and prediction records.

{md_table(runs,['kind','seed','best_epoch','epochs','frame_macro_f1','source_macro_f1','switch_rate'])}

Models are selected by validation **frame Macro-F1 on the exact same 1,065 target
identities**, the frozen intersection of valid GRU16/GRU32 targets. MLP uses all
10,517 eligible Train frames; GRUs use accepted Train windows with their recorded
stride/cap. This is a comparison of trained models, not reuse of a full-data model.
The same class labels, groups and matched prediction intervals apply to all rows.

{md_table(ranking,['kind','mean_frame_f1','std_frame_f1','mean_take_f1','mean_clip_f1','mean_switch_rate','parameters'])}

Clip/take Macro-F1 pools mean probabilities within camera clips/original takes.
Supplementary majority-vote clip/take accuracy is in `majority_vote_accuracy.csv`.
Raw switching counts use only adjacent original target frames in one clip; no
post-hoc class smoothing is used for the comparison.

Coverage on the full validation partition (seed 302; identical across seeds):

{md_table(coverage_table,['kind','predicted_frames','eligible_target_coverage','raw_frame_coverage','take_coverage'])}

GRU16 covers 1,427/1,894 eligible validation frames; GRU32 covers 1,065/1,894.
Common evaluation covers 12/13 validation takes and 21/26 camera clips. One
validation take has no accepted GRU32 targets: it is not hidden, and MLP's full
coverage result is separately saved. Warmup/rejection coverage prevents a filtered
GRU score from being presented as coverage of all difficult frames.

![Matched validation, coverage and switching](comparisons/validation_comparison.png)

![Actual matched validation learning curves](comparisons/learning_curves.png)

Fixed-six-class per-subject Macro-F1 is recorded for compatibility, but many subjects
have only one or two validation/test classes; their fixed-six scores include zeros
for absent labels. `subject_scores_present_classes.csv` additionally reports each
subject's F1 over its actually present ground-truth classes, with the class list.
Neither definition is an unseen-subject benchmark. Likewise the control's front
camera has no matched class-3 targets: its fixed-six camera Macro-F1 is 0.8333
despite 1.0 accuracy. Camera/label coverage is saved rather than silently dropping
absent classes.

### Failure analysis and uncertainty

At seed 302 on common validation targets, MLP has 42 backward-to-upright errors
and one backward-to-hand error. GRU16 adds 39 desk-to-hunched errors and has 32
backward-to-upright / 14 backward-to-hand errors. GRU32 also has 39 desk-to-hunched
errors, with 43 backward-to-upright / 21 backward-to-hand errors. GRUs did not
resolve the difficult pairs and switched more often. Low-confidence fractions and
ECE/Brier values are saved; lower GRU low-confidence frequency does not mean better
calibration when confident predictions are wrong.

Paired uncertainty resamples original source takes, averaging both camera views
and the three seeds within each take; frames are not counted as independent
subjects. `selection.json` contains descriptive 95% bootstrap intervals for mean
take-accuracy differences. These are development-selection summaries, not corrected
population confidence intervals. GRU16's interval includes zero; GRU32 is negative.
The promotion rule required >=0.01 matched mean F1 gain or >=25% flicker reduction
with <=0.005 F1 loss and no class recall drop >0.03. Both GRUs failed. No further
tuning followed the internal test.

## Once-only internal take test

**{TEST_LABEL}**

Winner/representative seed: fresh `mlp_control`, seed 302, best epoch 31. The
selection was frozen before test access. This checkpoint trained only on the 61
Train takes; neither old all-data model was scored on these held-out takes.

* Predictions: {internal['frames']} eligible frames, {internal['clips']} usable camera
  clips, {internal['source_takes']} original takes, 9 training-seen subjects.
* Frame accuracy: **{internal['frame_accuracy']:.4f}**; frame Macro-F1: **{internal['frame_macro_f1']:.4f}**.
* Clip Macro-F1: **{internal['clip_macro_f1']:.4f}**; take Macro-F1: **{internal['source_macro_f1']:.4f}**.
* Raw switching rate: {internal['switch_rate']:.4f}; ECE: {internal['calibration_ece']:.4f}.
* Evaluation count: **1**. Common-target frame metrics were derived from cached
  predictions only, without another forward pass, and are separately identified.

Most test errors are 105 hunched-to-desk predictions and 52 upright-to-sideways
predictions. Per-class precision/recall/F1, subject/camera results, coverage, loss,
confusion matrix and cached predictions are in `internal_test/`. The test did not
trigger any configuration or threshold changes. It measures seen-subject unseen-take
performance, not unseen-subject or independent external generalization.

![Once-only internal take-level confusion matrix](internal_test/confusion_matrix.png)

## Full-data candidate and real inference

MLP remains preferred. The existing full-data candidate already includes all 13
subjects and all 14,256 eligible observations with the original feature contract.
It was preserved without an unnecessary refit. The new take-control median best
epoch is 14, while the retained release uses the historical subject-development
four-epoch schedule; seen-subject take validation was not used to silently replace
that previous broader-development choice. No final full-data model accuracy is
claimed from these internal records, which that all-data model has already seen.

Selected PT: `{Path(final['path'])/'model_final.pt'}`.
Selected ONNX: `{Path(final['path'])/'model_final.onnx'}`.
Research GRU16 PT/ONNX: `{run/'gru16/deployment'}`.
Research GRU32 PT/ONNX: `{run/'gru32/deployment'}`.
Those research GRUs are train-partition checkpoints, not promoted all-data releases.
Each has architecture/feature/sequence contracts, labels, training lineage/history
and a deployment verification report. Full-data GRU refitting is implemented for a
future validation winner; it was not run for these losing candidates.

{md_table(cpu,['model','median_ms','p95_ms','max_probability_error','smoke_frames','smoke_predictions','smoke_fps'])}

ONNX uses fixed sequence lengths 16/32 with dynamic batch, tested at batch 1, 8
and representative real batches. Export warnings about generic RNN tracing are
retained in logs; real probability parity passed (<1e-5). Both loaders return six
finite probabilities in the official class order. Cached real Train/Validation
buffer replay checked 13,485 original frames per sequence length and exactly matched
offline context/target eligibility. Existing MLP v001 still loads and its hash is
unchanged. Every audited original dataset and previous release hash passed.

The end-to-end smoke used a **Train** video, CHANH label-1 cam01, never the internal
test. All three paths processed 73 frames with YOLO11n-Pose, ByteTrack, existing
person selection and causal normalization. GRU16 first predicts after 15 warmup
frames (58 predictions); GRU32 after 31 (42 predictions). These are functional
checks, not accuracy estimates; GRU16's aggregate prediction was wrong on this
smoke clip, another reason not to equate successful execution with classification
quality. End-to-end FPS includes cold-start/model/writer overhead and should not
be interpreted as a controlled speedup. Warmed ONNX CPU predict-call latency is
reported separately in milliseconds, excluding YOLO.

Temporal buffers belong to each PosePipeline/camera, never to the shared model.
Track/session changes, stream reconnect, long observation gaps and long missing
runs reset context. Missing-current-pose frames always abstain; stale probabilities
are not reused. Each prediction recomputes a causal rolling window with zero GRU
initial hidden state, matching training. No future frames, cross-camera hidden
state or new keypoint smoothing are used. Video GRU inference requires true capture
timestamps via an auto-detected camera sidecar or `--timestamps`; encoded FPS is
not silently substituted. Live readers use reception timestamps and are implemented
but were **not accessed or tested on real hardware**, as requested by the user.

## External testing, implementation and tests

External accuracy and live hardware tests are **PENDING**. The user confirmed no
new independent recordings exist and explicitly deferred webcam access. No camera
probe, network camera request, consent assumption, volunteer count or external
accuracy was invented. `REAL_WORLD_TEST_REPORT.md` records only that pending state.

`scripts/evaluate_real_world.py` provides a guarded future labeled-video workflow:
new subject IDs, explicit consent metadata, isolated dataset root, original capture
timestamps, and hashes rejecting copied training videos/duplicates. It reuses
the existing inference path for annotated preview, confidence/class overlay,
frame CSV/JSONL, clip aggregation, coverage, latency/FPS and external metrics.
An evaluation-start registry prevents silently rerunning the same frozen manifest
and model. Its `--help` and guards were tested; no external dataset evaluation ran.
The existing live CLI gains per-frame CSV/JSONL and per-stream GRU buffers; it
remains pending explicit later hardware authorization. No second GUI was created.

Created `shared/gru_model.py`, `shared/sequence_data.py`, `shared/take_gru_training.py`,
`scripts/evaluate_real_world.py`, and `tests/test_take_gru.py`. Extended
`scripts/train_poseforge.py` with `take_level` stages; added a GRU factory to
`shared/mlp_model.py`; added optional clipping/frame-selection/output-directory
hooks to `shared/real_training.py`; extended `scripts/export_onnx.py` and
`inference/pipeline.py`/`run.py` for sequence input, timestamps and exports. Original
MLP architectures and causal feature implementation remain unchanged. Previous
subject-wise and within-video implementations remain reproducible. No dependency
installation, raw extraction, relabeling or temporal embargo was performed.

Full regression: {full_summary}
Focused regression: {focused_summary}
The full suite was run without deselection. Its remaining known failure is the
Windows asset-migration symbolic-link privilege test (WinError 1314); no new test
was weakened or failure hidden. Collector GUI/file-based camera tests use synthetic
feeds, not real camera hardware. GRU forward/backward, causal prefixes, timing,
source grouping, matched coverage, state resets, no stale predictions, external
guards, legacy MLP and PT/ONNX round trips are tested. All execution logs and actual
PowerShell commands are saved in the run and `docs/TRAINING_WINDOWS.md`.

Remaining limitations: small number of independent takes, mostly static labels,
short/irregular captures, pose-empty views, reduced GRU startup coverage, limited
per-subject class coverage, and absent independent external/live validation.
Historical subject-wise Wide CV (0.8467) remains separate evidence about unseen
subjects. This new take split does not establish that capability. Neither geometry
nor prior rejected augmentation sweeps were repeated, and no external test was run.
'''
    real_world='''# PoseForge real-world testing status

**PENDING — no external accuracy or live hardware test was executed.**

The user confirmed that genuinely new labeled external recordings are not yet
available and explicitly deferred all live webcam testing. No physical camera
was opened or probed, no network camera was accessed, and volunteer consent was
not assumed. Camera accessibility is unknown because no hardware test was permitted.

The selected model remains the existing all-data MLP Wide candidate. Real trained
GRU-16/32 research checkpoints and their PT/ONNX paths passed software/real-video
functional smoke tests on an existing **training** take. Those smoke runs are not
independent external testing and do not populate external accuracy metrics.

Ready workflow: scripts/evaluate_real_world.py --help. It requires a newly
authorized isolated dataset root and a JSON manifest with dataset_role set to
independent_external_test, consent_confirmed true, and recordings containing
person_id, session_id, take_id, camera_id, label, video_path, timestamps_path.
Paths are relative to that isolated root. Existing participant IDs and copied
training-video hashes are rejected. No placeholder recordings are included.

Recommended future collection after consent: 2–3 genuinely new volunteers, all
six labels, two separate 5–10-second takes per volunteer/class where practical,
two views, and original capture timestamps. No participant or clip counts are
claimed as collected. The actual current external dataset is unavailable.

After explicit authorization, freeze the model, manifest, thresholds and hashes;
run labeled external evaluation once. Report coverage/abstentions, frame/clip/take
and subject/class/camera metrics, stability, latency and FPS. If failures drive
further tuning, those recordings become development data and another independent
external test is needed. Live inference without labels is only a functional check.

See docs/TRAINING_WINDOWS.md for actual tested commands and separately marked
future commands that have NOT been executed. External metrics remain null/pending.
'''
    report_path=root/'artifacts/reports/TAKE_LEVEL_GRU_TRAINING_REPORT.md'
    external_path=root/'artifacts/reports/REAL_WORLD_TEST_REPORT.md'
    if report_path.exists() or external_path.exists():raise FileExistsError('Do not overwrite previous reports')
    for relative in ['comparisons/validation_comparison.png','comparisons/learning_curves.png','internal_test/confusion_matrix.png']:
        text=text.replace(']('+relative+')',']('+str(run/relative).replace('\\','/')+')')
    report_path.write_text(text,encoding='utf-8');external_path.write_text(real_world,encoding='utf-8')
    (run/'TAKE_LEVEL_GRU_TRAINING_REPORT.md').write_text(text,encoding='utf-8')
    rt.write_json(run/'external_status.json',dict(status='PENDING',independent_recordings_available=False,live_testing_authorized=False,
                    cameras_accessed=False,external_accuracy_metrics=None,reason='Explicit user instruction: defer all external and live hardware testing.'),exclusive=True)
    # Shareable review has only code, metrics, plots and manifests; no raw videos,
    # raw keypoints, feature arrays or optimizer states. Research model bundles
    # are separate and explicitly marked unpromoted.
    package=run/'take_gru_review.zip'
    with ZipFile(package,'x',ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(run.rglob('*')):
            if not path.is_file() or 'processed' in path.relative_to(run).parts:continue
            if path.suffix in ['.md','.csv','.json','.png','.py','.log'] or path.name.endswith('.csv.gz'):
                z.write(path,path.relative_to(run).as_posix())
        z.write(external_path,'REAL_WORLD_TEST_REPORT.md');z.write(root/'docs/TRAINING_WINDOWS.md','docs/TRAINING_WINDOWS.md')
    for kind in ['gru16','gru32']:
        folder=run/kind/'deployment'
        with ZipFile(folder/'research_model_bundle.zip','x',ZIP_DEFLATED,compresslevel=6) as z:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in ['.pt','.onnx','.json','.csv']:z.write(path,path.name)
            z.writestr('STATUS.md','Research take-train-only GRU. Not promoted over MLP. No independent external accuracy is claimed.\n')
    rt.write_json(run/'completion_summary.json',dict(selected=final,selection=selection,
                   internal_test_count=internal['test_evaluation_count'],test_frame_f1=internal['frame_macro_f1'],
                   all_previous_data_and_release_hashes_unchanged=True,external_test_executed=False,live_camera_accessed=False,
                   report=str(report_path),real_world_report=str(external_path),review_package=str(package)),exclusive=True)
    return dict(training_report=str(report_path),real_world_report=str(external_path),review_package=str(package))
