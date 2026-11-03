"""Development-only improvement experiments. QBAO/test evaluation is forbidden."""
from __future__ import annotations

from copy import deepcopy
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
from shared.dataset_pipeline import COCO13, file_sha256
from shared.mlp_model import build_classifier_model
from shared.pose_features import FEATURE_VERSION
from shared.pose_geometry import GEOMETRY_VERSION, GEOMETRY_DIM, GEOMETRY_NAMES, geometry_features

LOG = logging.getLogger('poseforge.training')
EXCLUDED_PERSON = 'QBAO'
VARIANTS = {
    'control': dict(model='mlp_wide', feature_mode='base'),
    'geometry': dict(model='mlp_wide', feature_mode='geometry'),
    'residual': dict(model='mlp_residual', feature_mode='base'),
    'layernorm': dict(model='mlp_layernorm', feature_mode='base'),
    'targeted_arms': dict(model='mlp_wide', feature_mode='base', quality_gated_augmentation=True,
                         jitter_prob=.3, jitter_sigma=.0075, confidence_perturb_prob=.2,
                         occlusion_prob=.12, max_dropout_joints=1),
}


def candidate_config(name, seed=302):
    return {**rt.DEFAULT, 'model':'mlp_wide', **VARIANTS[name], 'seed':int(seed)}


def feature_contract(run,cfg):
    contract=rt.read_json(Path(run)/'feature_contract.json')
    contract.update(source_dataset_id=contract['dataset_id'],dataset_id=file_sha256(Path(run)/'processed/checksums.json'))
    if cfg.get('feature_mode')=='geometry':
        contract.update(version=GEOMETRY_VERSION,flatten_dim=GEOMETRY_DIM,
                        base_version=FEATURE_VERSION,geometry_names=GEOMETRY_NAMES,
                        geometry_layout='base52,values22,masks22',geometry_clip=[-4,4],
                        geometry_policy='Current measured 2D joints only; safe denominators; zero invalid values and explicit validity masks.')
    return contract


def ensure_development(people):
    if EXCLUDED_PERSON in set(people):
        raise ValueError('QBAO is prohibited in development training/evaluation')


def prepare(source, output, raw_root, baseline):
    source, raw_root, baseline = Path(source).resolve(), Path(raw_root).resolve(), Path(baseline).resolve()
    split = rt.read_json(source/'split_manifest.json')
    dev = sorted(split['train_people'] + split['validation_people'])
    ensure_development(dev)
    catalog = pd.read_csv(source/'catalog.csv')
    catalog = catalog[catalog.person_id.isin(dev)].copy()
    assert not catalog.person_id.eq(EXCLUDED_PERSON).any()
    assert set(catalog.person_id) == set(dev)
    run = rt.version_dir(output, 'dev_v')
    (run/'processed').mkdir()
    checks = rt.read_json(source/'processed/checksums.json')
    inventory = rt.read_json(source/'snapshot/source_inventory.json')
    selected_inventory, frames = {}, []
    for row in catalog.to_dict('records'):
        src = source/'processed'/f'{row["clip_id"]}.npz'
        if file_sha256(src) != checks[src.name]:
            raise RuntimeError('Original immutable processed data changed')
        shutil.copyfile(src, run/'processed'/src.name)
        pose_file = rt.pose_path(source/'snapshot', row)
        items = [json.loads(line) for line in pose_file.read_text(encoding='utf-8').splitlines()]
        with np.load(src, allow_pickle=False) as arrays:
            eligible = arrays['eligible'].copy()
            valid = arrays['x'][:,:,3].sum(1)
            confidence = arrays['x'][:,:,2].sum(1) / np.maximum(valid,1)
            ticks = arrays['capture_ns']
            delta = np.r_[0,np.diff(ticks)]
        for i,item in enumerate(items):
            reasons = []
            if item.get('keypoints') is None:
                reasons.append('no_selected_pose:'+str(item.get('status')))
            elif rt.uncertain_selection(item):
                reasons.append('uncertain_selected_person')
            if valid[i] < 4:
                reasons.append('fewer_than_four_measured_joints')
            if not eligible[i] and not reasons:
                reasons.append('existing_curated_or_camera_exclusion')
            frames.append(dict(clip_id=row['clip_id'], person=row['person_id'], label=int(row['label']),
                               camera=row['camera_id'], split=row['split'], frame_index=i,
                               eligible=bool(eligible[i]), exclusion_reason=';'.join(reasons),
                               manual_review=bool(reasons), valid_joints=int(valid[i]),
                               mean_valid_confidence=float(confidence[i]),
                               duplicate_timestamp=bool(i>0 and delta[i]==0),
                               gap_over_250ms=bool(delta[i]>250_000_000),
                               person_count=item.get('person_count',0)))
        video = rt.inside(raw_root,row['video_path'])
        paths = [video, rt.inside(raw_root,row['metadata_path']),
                 video.parent/f'{row["camera_id"]}_timestamps.csv',rt.pose_path(raw_root,row)]
        for file in paths:
            rel = file.relative_to(raw_root).as_posix()
            if file_sha256(file) != inventory[rel]['sha256']:
                raise RuntimeError('Live development source differs from frozen data: '+rel)
            selected_inventory[rel] = inventory[rel]
    q = pd.DataFrame(frames)
    assert q.eligible.sum() == catalog.usable_frames.sum()
    q.to_csv(run/'quality_frames.csv.gz',index=False,compression='gzip')
    for dimensions in [('label',),('camera',),('person',),('person','label','camera')]:
        q.groupby(list(dimensions)).agg(frames=('eligible','size'),eligible=('eligible','sum'),
                  mean_valid_joints=('valid_joints','mean'),mean_valid_confidence=('mean_valid_confidence','mean'),
                  review_frames=('manual_review','sum')).reset_index().to_csv(run/f'quality_{"_".join(dimensions)}.csv',index=False)
    exclusions = q[~q.eligible].groupby(['clip_id','person','label','camera','exclusion_reason']).size().reset_index(name='frames')
    exclusions.to_csv(run/'exclusions_and_review.csv',index=False)
    catalog.to_csv(run/'catalog.csv',index=False)
    contract=rt.read_json(source/'feature_contract.json')
    rt.write_json(run/'feature_contract.json',contract,exclusive=True)
    rt.write_json(run/'development_split.json',dict(train_people=split['train_people'],validation_people=split['validation_people'],
                  development_people=dev,excluded_people=[EXCLUDED_PERSON],test_policy='NO TEST PREDICTIONS OR TEST METRIC READS'),exclusive=True)
    rt.write_json(run/'source_inventory.json',selected_inventory,exclusive=True)
    rt.write_json(run/'baseline_preservation.json',dict(path=str(baseline),
                  files={p.relative_to(baseline).as_posix():file_sha256(p) for p in baseline.rglob('*') if p.is_file()}),exclusive=True)
    rt.write_json(run/'frozen_manifest_checksums.json',{n:file_sha256(run/n) for n in ['catalog.csv','feature_contract.json','development_split.json']},exclusive=True)
    rt.seal(run/'processed')
    shuffled=np.random.default_rng(4302).permutation(dev).tolist()
    folds=[dict(fold=i,validation_people=sorted(shuffled[i*4:(i+1)*4]),
                train_people=sorted(set(dev)-set(shuffled[i*4:(i+1)*4]))) for i in range(3)]
    rt.write_json(run/'protocol.json',dict(source_run=str(source),raw_root=str(raw_root),development_people=dev,
                original_validation_people=split['validation_people'],excluded_people=[EXCLUDED_PERSON],
                candidates={name:candidate_config(name) for name in VARIANTS},screen_seed=302,
                confirmation_seeds=[302,303,304],cv_seeds=[302,303],folds=folds,
                cv_policy='Fixed epoch schedules from original-split development confirmation; no fold-validation checkpoint selection. CV is development evidence, not an independent test.',
                promotion=dict(matched_subject_f1_gain=.01,cv_subject_f1_gain=.01,min_positive_folds=2,
                  max_camera_f1_drop=.02,max_class_recall_drop=.03,max_worst_subject_drop=.02,
                  alternative_robustness_gain=.02,max_clean_drop_for_robustness=.005),
                augmentation='Quality gated: >=10 joints, mean measured confidence >=0.70; one forearm joint max; no head/torso dropout or flip.',
                policy='Reuses exactly the original eligible-frame mask. No guessed selections, pose recovery, curation relabeling, or test use.'),exclusive=True)
    empty = catalog.loc[catalog.pose17_frames==0,'clip_id'].tolist()
    quality=dict(subjects=len(dev),clips=len(catalog),source_takes=int(catalog.source_group_id.nunique()),
                 frames=len(q),eligible=int(q.eligible.sum()),excluded=int((~q.eligible).sum()),
                 pose_empty_clips=empty,excluded_people=[EXCLUDED_PERSON],
                 timestamp_duplicates=int(q.duplicate_timestamp.sum()),large_timestamp_gaps=int(q.gap_over_250ms.sum()),
                 mean_clip_duration_s=float(catalog.capture_span_s.mean()),
                 policy='Existing conservative eligibility unchanged. Missing frames/confidences retained; all no-selection and uncertain observations require review.',
                 new_snapshot_reason='Isolate the 12 development subjects physically; QBAO arrays are not copied or loaded.')
    rt.write_json(run/'dataset_quality.json',quality,exclusive=True)
    rt.write_json(run/'experiment_protocol_checksums.json',{name:file_sha256(run/name) for name in
                  ['protocol.json','development_split.json','source_inventory.json','baseline_preservation.json']},exclusive=True)
    (run/'DATA_QUALITY.md').write_text('# Development dataset quality\n\n'+json.dumps(quality,indent=2)+'\n\nSee exclusions_and_review.csv and quality_*.csv. The prior full dataset had 16 pose-empty camera clips; 15 belong to development. No recovery is assumed. Strong correlation within takes is handled through subject-wise folds and subject-level uncertainty; source balancing already failed prior validation and is not repeated. Capture order and original masks are unchanged.\n',encoding='utf-8')
    LOG.info('Prepared development-only %s: %s',run,quality)
    return run


class DevelopmentData(rt.TrainingData):
    def __init__(self,run,device,cfg,fold=None):
        lock=Path(run)/'experiment_protocol_checksums.json'
        if lock.exists():
            for name,digest in rt.read_json(lock).items():
                if file_sha256(Path(run)/name)!=digest:raise RuntimeError('Frozen improvement protocol changed')
        super().__init__(run,['train','val'],device)
        ensure_development(c['person'] for c in self.clips)
        self.cfg=cfg
        self.input_dim=GEOMETRY_DIM if cfg.get('feature_mode')=='geometry' else 52
        if fold is not None:
            val=set(fold['validation_people']);train=set(fold['train_people'])
            assert not val&train and val|train=={c['person'] for c in self.clips}
            for c in self.clips:
                c['split']='val' if c['person'] in val else 'train'

    def model_features(self,x):
        return geometry_features(x) if self.input_dim==GEOMETRY_DIM else x

    def training_features(self,x,cfg,generator):
        if cfg.get('quality_gated_augmentation'):
            a=x.reshape(-1,13,4)
            n=a[:,:,3].sum(1)
            good=(n>=10)&(a[:,:,2].sum(1)/n.clamp(min=1)>=.70)
            x=x.clone()
            x[good]=rt.augment_pose13_torch(x[good],cfg,generator)
        else:
            x=rt.augment_pose13_torch(x,cfg,generator)
        return self.model_features(x)


def model_from_result(run,result,device):
    name=result['name'];cfg=result['config']
    model=build_classifier_model(cfg['model'],cfg['dropout'],GEOMETRY_DIM if cfg.get('feature_mode')=='geometry' else 52).to(device)
    saved=torch.load(Path(run)/'experiments'/name/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    return model.eval()


def diagnostics(run,result,device,fold=None):
    folder=Path(run)/'experiments'/result['name']
    cfg=result['config']; data=DevelopmentData(run,device,cfg,fold)
    model=model_from_result(run,result,device)
    val=[c for c in data.clips if c['split']=='val']
    stress={}
    for name,joints in [('left_arm',[7,9]),('right_arm',[8,10]),('nose_missing',[0])]:
        original=data.xt
        altered=original.clone().reshape(-1,13,4)
        altered[:,joints,:]=0
        data.xt=altered.reshape(-1,52)
        score,_=rt.evaluate(model,data,val)
        stress[name]={k:score[k] for k in ['subject_macro_f1','frame_macro_f1','clip_macro_f1']}
        data.xt=original
    sample=data.xt[data.ids(val)[:256]][:1]
    prepared=data.model_features(sample)
    with torch.inference_mode():
        for _ in range(30):model(prepared)
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(300):model(prepared)
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        ms=(time.perf_counter()-start)*1000/300
    size=sum(p.numel() for p in model.parameters())
    report=dict(parameter_count=size,cuda_batch1_ms=ms,latency_scope='Classifier only, warmed 300 calls, batch 1; features precomputed; CUDA synchronized around loop.',
                input_dim=data.input_dim,stress=stress,
                mean_arm_occlusion_f1=float(np.mean([stress[k]['subject_macro_f1'] for k in ['left_arm','right_arm']])))
    rt.write_json(folder/'robustness_latency.json',report)
    return report


def execute(run,name,seed,device,resume=False,prefix='',fold=None,fixed_epochs=None):
    cfg=candidate_config(name,seed)
    fullname=f'{prefix}{name}_seed{seed}'
    data=DevelopmentData(run,device,cfg,fold)
    result=rt.run_candidate(run,cfg,device,fullname,resume,data=data,fixed_epochs=fixed_epochs)
    contract_path=Path(run)/'experiments'/fullname/'feature_contract.json'
    if not contract_path.exists():rt.write_json(contract_path,feature_contract(run,cfg),exclusive=True)
    if not (Path(run)/'experiments'/fullname/'robustness_latency.json').exists():
        diagnostics(run,result,device,fold)
    return result


def summarize(run):
    records=[]
    for path in sorted((Path(run)/'experiments').glob('*/result.json')):
        r=rt.read_json(path)
        if 'metrics' not in r or r['name'].startswith('smoke'):continue
        extra=rt.read_json(path.parent/'robustness_latency.json') if (path.parent/'robustness_latency.json').exists() else {}
        records.append(dict(experiment=r['name'],model=r['config']['model'],feature_mode=r['config'].get('feature_mode','base'),
                    seed=r['config']['seed'],best_epoch=r['best_epoch'],epochs=r['epochs_completed'],
                    **{k:v for k,v in r['metrics'].items() if isinstance(v,(int,float))},
                    **{k:v for k,v in extra.items() if isinstance(v,(int,float))}))
    df=pd.DataFrame(records)
    df.to_csv(Path(run)/'comparison.csv',index=False)
    return df


def screen(run,device,resume=False):
    cfg={**candidate_config('control'),'epochs':5,'patience':5}
    smoke=rt.run_candidate(run,cfg,device,'smoke_control',resume,data=DevelopmentData(run,device,cfg))
    history=pd.read_csv(Path(run)/'experiments/smoke_control/history.csv')
    if history.train_loss.iloc[-1]>=history.train_loss.iloc[0]:raise RuntimeError('Smoke loss failed to decrease')
    for name in VARIANTS:
        execute(run,name,302,device,resume)
        summarize(run)
    # The original matched control should reproduce exactly before comparison.
    source=Path(rt.read_json(Path(run)/'protocol.json')['source_run'])
    original=rt.read_json(source/'experiments/baseline_mlp_wide/result.json')['metrics']
    current=rt.read_json(Path(run)/'experiments/control_seed302/result.json')['metrics']
    if abs(original['subject_macro_f1']-current['subject_macro_f1'])>1e-7:
        raise RuntimeError('Matched control did not reproduce original frozen validation score')


def confirm(run,device,resume=False):
    df=summarize(run)
    first=df[~df.experiment.str.startswith(('cv','smoke')) & df.seed.eq(302)]
    shortlist=first[~first.experiment.str.startswith('control_')].sort_values('subject_macro_f1',ascending=False).head(2)
    names=['control']+[s.rsplit('_seed',1)[0] for s in shortlist.experiment]
    for name in names:
        for seed in [303,304]:
            execute(run,name,seed,device,resume)
            summarize(run)
    records=[]
    for name in names:
        for seed in [302,303,304]:
            r=rt.read_json(Path(run)/f'experiments/{name}_seed{seed}/result.json')
            extra=rt.read_json(Path(run)/f'experiments/{name}_seed{seed}/robustness_latency.json')
            records.append(dict(candidate=name,seed=seed,f1=r['metrics']['subject_macro_f1'],
                                clip_f1=r['metrics']['clip_macro_f1'],epoch=r['best_epoch'],
                                arm_occlusion_f1=extra['mean_arm_occlusion_f1']))
    seed_table=pd.DataFrame(records)
    seed_table.to_csv(Path(run)/'seed_confirmation.csv',index=False)
    ranking=seed_table.groupby('candidate').agg(mean_f1=('f1','mean'),std_f1=('f1','std'),
              mean_clip_f1=('clip_f1','mean'),median_epoch=('epoch','median'),arm_occlusion_f1=('arm_occlusion_f1','mean')).reset_index()
    ranking=ranking.sort_values('mean_f1',ascending=False)
    ranking.to_csv(Path(run)/'confirmed_ranking.csv',index=False)
    best=ranking[ranking.candidate!='control'].iloc[0].candidate
    plan=dict(candidates=['control',str(best)],schedules={r['candidate']:max(1,int(round(r['median_epoch']))) for r in ranking.to_dict('records')},
              policy='Top confirmed non-control versus control; CV schedules frozen before fold training.')
    path=Path(run)/'cv_plan.json'
    if path.exists() and rt.read_json(path)!=plan:raise RuntimeError('CV plan already frozen')
    rt.write_json(path,plan)
    return ranking.to_dict('records')


def cross_validate(run,device,resume=False):
    protocol=rt.read_json(Path(run)/'protocol.json');plan=rt.read_json(Path(run)/'cv_plan.json')
    lock=Path(run)/'cv_plan_sha256.json'
    digest=file_sha256(Path(run)/'cv_plan.json')
    if lock.exists() and rt.read_json(lock)['sha256']!=digest:raise RuntimeError('Frozen CV plan changed')
    if not lock.exists():rt.write_json(lock,dict(sha256=digest),exclusive=True)
    for fold in protocol['folds']:
        for seed in protocol['cv_seeds']:
            for name in plan['candidates']:
                execute(run,name,seed,device,resume,prefix=f'cv{fold["fold"]}_',fold=fold,fixed_epochs=plan['schedules'][name])
                summarize(run)
    return select(run)


def cross_validate_robustness(run,device,resume=False):
    run=Path(run);protocol=rt.read_json(run/'protocol.json');plan=rt.read_json(run/'cv_plan.json')
    ranking=pd.read_csv(run/'confirmed_ranking.csv').set_index('candidate')
    baseline=ranking.loc['control']
    eligible=ranking[(ranking.index!='control') &
                     (ranking.mean_f1>=baseline.mean_f1-.005) &
                     (ranking.arm_occlusion_f1>=baseline.arm_occlusion_f1+.02)]
    if eligible.empty:return dict(status='No robustness follow-up justified')
    candidate=str(eligible.sort_values('arm_occlusion_f1',ascending=False).index[0])
    followup=dict(candidate=candidate,schedule=plan['schedules'][candidate],seeds=protocol['cv_seeds'],
                  folds=protocol['folds'],unchanged_original_plan_sha256=file_sha256(run/'cv_plan.json'),
                  reason='Matched three-seed arm-occlusion gain >=0.02 with clean mean within 0.005 of control; test this confirmed alternative under the same pre-existing promotion gates. No new hyperparameters.',
                  matched_robustness_gain=float(ranking.loc[candidate,'arm_occlusion_f1']-baseline.arm_occlusion_f1))
    path=run/'cv_robustness_plan.json'
    if path.exists() and rt.read_json(path)!=followup:raise RuntimeError('Robustness follow-up plan changed')
    if not path.exists():rt.write_json(path,followup,exclusive=True)
    for fold in protocol['folds']:
        for seed in protocol['cv_seeds']:
            execute(run,candidate,seed,device,resume,prefix=f'cv{fold["fold"]}_',fold=fold,fixed_epochs=followup['schedule'])
            summarize(run)
    return select(run,candidate)


def select(run,candidate=None):
    run=Path(run);protocol=rt.read_json(run/'protocol.json');plan=rt.read_json(run/'cv_plan.json')
    candidate=candidate or next(n for n in plan['candidates'] if n!='control')
    pair=['control',candidate]
    rows=[];per_subject=[];class_rows=[];camera_rows=[]
    for fold in protocol['folds']:
        for seed in protocol['cv_seeds']:
            for name in pair:
                path=run/f'experiments/cv{fold["fold"]}_{name}_seed{seed}'
                r=rt.read_json(path/'result.json');m=r['metrics'];stress=rt.read_json(path/'robustness_latency.json')
                rows.append(dict(candidate=name,fold=fold['fold'],seed=seed,f1=m['subject_macro_f1'],clip_f1=m['clip_macro_f1'],arm_f1=stress['mean_arm_occlusion_f1']))
                per_subject.extend(dict(candidate=name,fold=fold['fold'],seed=seed,**p) for p in m['per_subject'])
                class_rows.extend(dict(candidate=name,fold=fold['fold'],seed=seed,label=int(k),recall=m['classification_report'][str(k)]['recall']) for k in range(6))
                camera_rows.extend(dict(candidate=name,fold=fold['fold'],seed=seed,**p) for p in m['per_camera'])
    folds=pd.DataFrame(rows);subjects=pd.DataFrame(per_subject)
    folds.to_csv(run/f'cross_validation_{candidate}.csv',index=False);subjects.to_csv(run/f'cv_subject_metrics_{candidate}.csv',index=False)
    subject_means=subjects.groupby(['person','candidate']).macro_f1.mean().unstack()
    delta=subject_means[candidate]-subject_means.control
    rng=np.random.default_rng(4302)
    boot=delta.to_numpy()[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    mean_cv=folds.groupby('candidate').f1.mean();arm_cv=folds.groupby('candidate').arm_f1.mean()
    fold_means=folds.groupby(['fold','candidate']).f1.mean().unstack()
    positive_folds=int((fold_means[candidate]>fold_means.control).sum())
    ranking=pd.read_csv(run/'confirmed_ranking.csv').set_index('candidate')
    matched_gain=float(ranking.loc[candidate,'mean_f1']-ranking.loc['control','mean_f1'])
    cv_gain=float(mean_cv[candidate]-mean_cv.control)
    arm_gain=float(arm_cv[candidate]-arm_cv.control)
    camera=pd.DataFrame(camera_rows).groupby(['camera','candidate']).macro_f1.mean().unstack()
    classes=pd.DataFrame(class_rows).groupby(['label','candidate']).recall.mean().unstack()
    min_camera=float((camera[candidate]-camera.control).min())
    min_class=float((classes[candidate]-classes.control).min())
    guard=protocol['promotion']
    no_major_drop=(min_camera>=-guard['max_camera_f1_drop'] and min_class>=-guard['max_class_recall_drop']
                   and float(delta.min())>=-guard['max_worst_subject_drop'])
    clean=(matched_gain>=guard['matched_subject_f1_gain'] and cv_gain>=guard['cv_subject_f1_gain'] and positive_folds>=2)
    robust=(arm_gain>=guard['alternative_robustness_gain'] and matched_gain>=-guard['max_clean_drop_for_robustness']
            and cv_gain>=-guard['max_clean_drop_for_robustness'] and positive_folds>=2)
    promoted=bool((clean or robust) and no_major_drop)
    selection=dict(candidate=candidate,recommended_candidate=candidate if promoted else 'control',promoted=promoted,
                   matched_validation_gain=matched_gain,cv_gain=cv_gain,cv_arm_occlusion_gain=arm_gain,
                   positive_folds=positive_folds,paired_subject_bootstrap_ci95=ci,bootstrap_unit='subject; two seeds averaged per subject; 12 development subjects',
                   statistically_convincing=bool(ci[0]>0),practically_convincing=bool(promoted),
                   min_subject_delta=float(delta.min()),min_camera_delta=min_camera,min_class_recall_delta=min_class,
                   clean_gain_gate=bool(clean),robustness_gate=bool(robust),non_degradation_gate=bool(no_major_drop),
                   final_epochs=plan['schedules'][candidate if promoted else 'control'],final_seed=10302,
                   test_accessed=False,external_test_executed=False)
    rt.write_json(run/f'selection_{candidate}.json',selection)
    assessments=[rt.read_json(p) for p in run.glob('selection_*.json')]
    candidates_that_pass=[s for s in assessments if s['promoted']]
    chosen=max(candidates_that_pass or assessments,key=lambda s:s['matched_validation_gain'])
    decision={**chosen,'candidate_assessments':assessments}
    rt.write_json(run/'selection.json',decision)
    # All completed CV candidates, without duplicating control rows.
    import re
    all_rows=[];all_subjects=[]
    for path in sorted((run/'experiments').glob('cv*/result.json')):
        match=re.fullmatch(r'cv(\d+)_(.+)_seed(\d+)',path.parent.name)
        if not match:continue
        f,name,seed=match.groups();m=rt.read_json(path)['metrics'];extra=rt.read_json(path.parent/'robustness_latency.json')
        all_rows.append(dict(candidate=name,fold=int(f),seed=int(seed),f1=m['subject_macro_f1'],clip_f1=m['clip_macro_f1'],arm_f1=extra['mean_arm_occlusion_f1']))
        all_subjects.extend(dict(candidate=name,fold=int(f),seed=int(seed),**p) for p in m['per_subject'])
    pd.DataFrame(all_rows).to_csv(run/'cross_validation.csv',index=False)
    pd.DataFrame(all_subjects).to_csv(run/'cv_subject_metrics.csv',index=False)
    return decision


def error_analysis(run):
    run=Path(run)
    folder=run/'experiments/control_seed302'
    predictions=pd.read_csv(folder/'frame_predictions.csv.gz')
    quality=pd.read_csv(run/'quality_frames.csv.gz')
    # Predictions are in original frame order after the frozen eligibility mask.
    eligible=quality[quality.eligible & quality.split.eq('val')].copy()
    assert len(eligible)==len(predictions)
    merged=[]
    for cid,part in predictions.groupby('clip_id',sort=False):
        q=eligible[eligible.clip_id==cid].sort_values('frame_index').reset_index(drop=True)
        part=part.reset_index(drop=True)
        assert len(q)==len(part)
        part['valid_joints']=q.valid_joints;part['mean_valid_confidence']=q.mean_valid_confidence
        merged.append(part)
    pred=pd.concat(merged,ignore_index=True)
    pred['correct']=pred.label.eq(pred.pred)
    pred['low_confidence']=pred.confidence<.7
    pred['visibility_bin']=pd.cut(pred.valid_joints,[0,9,11,13],labels=['4-9','10-11','12-13'])
    pred.to_csv(run/'validation_error_records.csv.gz',index=False,compression='gzip')
    pred[~pred.correct].groupby(['person','camera','label','pred']).size().reset_index(name='errors').to_csv(run/'error_pairs.csv',index=False)
    pred.groupby('visibility_bin',observed=True).agg(frames=('correct','size'),accuracy=('correct','mean'),confidence=('confidence','mean')).to_csv(run/'missing_joint_sensitivity.csv')
    pred.groupby(['label','low_confidence']).agg(frames=('correct','size'),accuracy=('correct','mean')).to_csv(run/'low_confidence_analysis.csv')
    text='# Development error analysis\n\n'
    text+='Original frozen validation only; no QBAO data. Dominant errors are VMTRI hunch-to-desk (98 frames) and backward-to-upright (42 frames), plus 4 TRUONG desk-to-backward frames. Nose observations are present in the dominant failure clips, so missingness alone does not explain errors. The static 2D geometry and subject style overlap motivate explicit geometry and normalized/residual architectures.\n\n'
    text+='Class 4 remains a single left/right class. Class 4 and class 5 have no control-validation errors; partial arm movements are not separately labeled. Arm-occlusion stress is a synthetic development diagnostic, not a real external evaluation. No posture/person relabeling follows model errors.\n\n'
    text+=pred[~pred.correct].groupby(['person','camera','label','pred']).size().to_string()+'\n'
    (run/'ERROR_ANALYSIS.md').write_text(text,encoding='utf-8')


def finalize(run,device,resume=False):
    """Fresh 12-subject development fit. Never invokes the old test finalizer."""
    run=Path(run).resolve()
    marker=run/'final_development.json'
    if marker.exists():return rt.read_json(marker)
    selection=rt.read_json(run/'selection.json')
    name=selection['recommended_candidate']
    cfg=candidate_config(name,selection['final_seed'])
    data=DevelopmentData(run,device,cfg)
    ensure_development(c['person'] for c in data.clips)
    result=rt.run_candidate(run,cfg,device,'final_development_fit',resume,final_epochs=selection['final_epochs'],data=data)
    if selection['promoted']:
        folder=rt.version_dir(run.parents[1]/'models'/cfg['model'])
    else:
        # An unproven experiment is kept in the research run, not installed over
        # or alongside the recommended production baseline as a promoted release.
        folder=rt.version_dir(run/'unpromoted_development_model')
    contract=feature_contract(run,cfg)
    labels=rt.read_json(Path(rt.read_json(run/'protocol.json')['source_run'])/'snapshot/config/labels.json')
    names=[r['name'] for r in sorted(labels['labels'],key=lambda r:r['id'])]
    trained=torch.load(run/'experiments/final_development_fit/trained.pt',map_location='cpu',weights_only=True)
    people=sorted({c['person'] for c in data.clips})
    trained.update(stage='final',feature_contract=contract,label_names=names,train_people=people,
                   dataset_id=file_sha256(run/'processed/checksums.json'),external_test_pending=True)
    rt.atomic_torch_save(trained,folder/'model_final.pt')
    rt.write_json(folder/'feature_contract.json',contract,exclusive=True)
    rt.write_json(folder/'model_config.json',cfg,exclusive=True)
    rt.write_json(folder/'label_map.json',labels,exclusive=True)
    rt.write_json(folder/'split_manifest.json',rt.read_json(run/'development_split.json'),exclusive=True)
    summary=dict(selection=selection,train_people=people,eligible_frames=int(data.valid.sum()),
                 seed=cfg['seed'],epochs=result['best_epoch'],external_test_executed=False,
                 test_metrics=None,promoted=selection['promoted'],checkpoint_sha256=file_sha256(folder/'model_final.pt'),
                 status='promoted_development_release' if selection['promoted'] else 'research_reproduction_only; retain mlp_wide/v001')
    rt.write_json(folder/'training_summary.json',summary,exclusive=True)
    shutil.copyfile(run/'experiments/final_development_fit/history.csv',folder/'history.csv')
    shutil.copyfile(run/'experiments/final_development_fit/learning_curves.png',folder/'learning_curves.png')
    result=dict(path=str(folder),promoted=selection['promoted'],recommended_model=str(folder) if selection['promoted'] else rt.read_json(run/'baseline_preservation.json')['path'])
    rt.write_json(marker,result,exclusive=True)
    return result


def assert_preserved(run):
    run=Path(run)
    baseline=rt.read_json(run/'baseline_preservation.json')
    for rel,sha in baseline['files'].items():
        if file_sha256(Path(baseline['path'])/rel)!=sha:raise RuntimeError('Baseline model modified')
    root=Path(rt.read_json(run/'protocol.json')['raw_root'])
    for rel,value in rt.read_json(run/'source_inventory.json').items():
        if file_sha256(root/rel)!=value['sha256']:raise RuntimeError('Original development source modified')
    rt.verify_seal(run/'processed')
    return dict(baseline_files_unchanged=len(baseline['files']),development_source_files_unchanged=len(rt.read_json(run/'source_inventory.json')),
                immutable_development_features=True,test_predictions_executed=False)


def deploy(run):
    from scripts.export_onnx import export
    from inference.pipeline import PostureClassifier
    import subprocess
    import sys
    run=Path(run).resolve(); final=rt.read_json(run/'final_development.json')
    folder=Path(final['path']);cfg=rt.read_json(folder/'model_config.json')
    if not (folder/'model.onnx').exists():export(folder)
    data=DevelopmentData(run,'cpu',cfg)
    base=data.x[data.ids(data.clips)]
    features=geometry_features(base) if cfg['feature_mode']=='geometry' else base
    ckpt=torch.load(folder/'model_final.pt',map_location='cpu',weights_only=True)
    model=build_classifier_model(cfg['model'],cfg['dropout'],features.shape[1]).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    onnx=PostureClassifier(folder,'onnx');pt=PostureClassifier(folder,'pt')
    probability_error=0.;logit_error=0.
    with torch.inference_mode():
        for start in range(0,len(features),1024):
            chunk=features[start:start+1024]
            a=model(torch.from_numpy(chunk));b=onnx.session.run(['logits'],{'features':chunk})[0]
            logit_error=max(logit_error,float(np.abs(a.numpy()-b).max()))
            probability_error=max(probability_error,float(np.abs(a.softmax(1).numpy()-torch.from_numpy(b).softmax(1).numpy()).max()))
    if probability_error>1e-5:raise RuntimeError('Real development ONNX probability parity failed')
    for x in base[:32]:np.testing.assert_allclose(pt.predict(x),onnx.predict(x),atol=1e-5,rtol=1e-5)
    # Existing production version remains readable and numerically equivalent.
    baseline=Path(rt.read_json(run/'baseline_preservation.json')['path'])
    oldpt=PostureClassifier(baseline,'pt');oldonnx=PostureClassifier(baseline,'onnx')
    for x in base[:32]:np.testing.assert_allclose(oldpt.predict(x),oldonnx.predict(x),atol=1e-5,rtol=1e-5)
    reproduction_identical=None
    if not final['promoted']:
        reproduction_identical=all(torch.equal(pt.model.state_dict()[key],value)
                                   for key,value in oldpt.model.state_dict().items())
        if not reproduction_identical:raise RuntimeError('Fresh baseline reproduction differs from preserved v001 weights')
    # Verify experimental NumPy deployment and Torch training geometry on real
    # development observations even when the geometry candidate is rejected.
    numpy_geometry=geometry_features(base)
    torch_geometry=geometry_features(torch.from_numpy(base)).numpy()
    geometry_error=float(np.abs(numpy_geometry-torch_geometry).max())
    np.testing.assert_allclose(numpy_geometry,torch_geometry,atol=1e-6,rtol=1e-6)
    latency=[]
    for name,classifier in [('final_development',onnx),('baseline_v001',oldonnx)]:
        for x in base[:50]:classifier.predict(x)
        ms=[]
        for i in range(1000):
            start=time.perf_counter_ns();p=classifier.predict(base[i%len(base)])
            ms.append((time.perf_counter_ns()-start)/1e6)
            assert p.shape==(6,) and np.isfinite(p).all() and np.isclose(p.sum(),1)
        latency.append(dict(model=name,backend='ONNX CPU',batch_size=1,calls=1000,median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),
                            scope='Predict call including geometry if selected, softmax and Python overhead; excludes pose extraction.'))
    pd.DataFrame(latency).to_csv(run/'deployment_latency.csv',index=False)
    rt.write_json(folder/'onnx_real_parity.json',dict(verified=True,real_development_frames=len(features),
                   max_abs_probability_difference=probability_error,max_abs_logit_difference=logit_error,
                   tolerance=1e-5,original_v001_loads=True,feature_contract=ckpt['feature_contract']['version'],
                   baseline_reproduction_weights_identical=reproduction_identical,
                   real_geometry_numpy_torch_max_abs_error=geometry_error,
                   test_predictions_executed=False))
    # A development video smoke, explicitly separate from external evaluation.
    root=run.parents[2]
    raw=Path(rt.read_json(run/'protocol.json')['raw_root'])
    source=raw/'recordings/CHANH/session_20261027_1124/label_0/take_001/cam01_raw.mp4'
    command=[sys.executable,str(root/'inference/run.py'),'--source',str(source),'--mode','video',
             '--model',str(folder),'--pose',str(root/'artifacts/pretrained/yolo11n-pose.pt'),
             '--device','cuda:0','--backend','onnx','--cls-device','cpu','--headless','--name','dev_improvement_e2e']
    marker=run/'inference_smoke.json'
    if not marker.exists():
        completed=subprocess.run(command,cwd=root,text=True,encoding='utf-8',capture_output=True)
        (run/'inference_smoke.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        if completed.returncode:raise RuntimeError('Development video inference smoke failed; see log')
        output=max((root/'artifacts/inference').glob('dev_improvement_e2e_v*/frames.csv'),key=lambda p:p.stat().st_mtime_ns)
        records=pd.read_csv(output)
        rt.write_json(marker,dict(command=command,frames=len(records),classified=int(records.class_id.notna().sum()),
                      output=str(output.parent),external_test=False),exclusive=True)
    preservation=assert_preserved(run)
    rt.write_json(run/'preservation_checks.json',preservation)
    return dict(onnx_probability_error=probability_error,preservation=preservation,final=final)


def profile_cpu(run):
    run=Path(run);rows=[]
    torch.set_num_threads(1)
    for name in VARIANTS:
        result=rt.read_json(run/f'experiments/{name}_seed302/result.json')
        model=model_from_result(run,result,'cpu')
        data=DevelopmentData(run,'cpu',result['config'])
        x=data.model_features(data.xt[data.ids([c for c in data.clips if c['split']=='val'])[:1]])
        with torch.inference_mode():
            for _ in range(100):model(x)
            timings=[]
            for _ in range(1000):
                start=time.perf_counter_ns();model(x);timings.append((time.perf_counter_ns()-start)/1e6)
        rows.append(dict(candidate=name,input_dim=data.input_dim,parameters=sum(p.numel() for p in model.parameters()),
                         cpu_batch1_median_ms=float(np.median(timings)),cpu_batch1_p95_ms=float(np.quantile(timings,.95)),
                         scope='PyTorch CPU, one thread, 100 warmup + 1000 batch-one calls; classifier only, features precomputed.'))
    df=pd.DataFrame(rows);df.to_csv(run/'parameter_latency_comparison.csv',index=False)
    return df


def markdown_table(df,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in df[columns].to_dict('records'):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r.values())+' |')
    return '\n'.join(lines)


def report(run):
    run=Path(run).resolve();root=run.parents[2]
    quality=rt.read_json(run/'dataset_quality.json');protocol=rt.read_json(run/'protocol.json')
    selection=rt.read_json(run/'selection.json');final=rt.read_json(run/'final_development.json')
    preservation=assert_preserved(run)
    error_analysis(run)
    latency=profile_cpu(run)
    df=summarize(run)
    first=df[df.seed.eq(302)&~df.experiment.str.startswith('cv')].copy()
    first.to_csv(run/'architecture_feature_ablation.csv',index=False)
    ranking=pd.read_csv(run/'confirmed_ranking.csv')
    cv=pd.read_csv(run/'cross_validation.csv')
    cv_summary=cv.groupby('candidate').agg(subject_macro_f1=('f1','mean'),clip_macro_f1=('clip_f1','mean'),arm_occlusion_f1=('arm_f1','mean')).reset_index()
    cv_summary.to_csv(run/'cv_comparison.csv',index=False)
    all_class=[];all_camera=[]
    import re
    for path in sorted((run/'experiments').glob('cv*/result.json')):
        match=re.fullmatch(r'cv(\d+)_(.+)_seed(\d+)',path.parent.name)
        if not match:continue
        fold,name,seed=match.groups();m=rt.read_json(path)['metrics']
        for label in range(6):
            all_class.append(dict(candidate=name,fold=int(fold),seed=int(seed),label=label,
                                  **m['classification_report'][str(label)]))
        all_camera.extend(dict(candidate=name,fold=int(fold),seed=int(seed),**v) for v in m['per_camera'])
    pd.DataFrame(all_class).to_csv(run/'cv_per_class_metrics.csv',index=False)
    pd.DataFrame(all_camera).to_csv(run/'cv_camera_metrics.csv',index=False)
    assessments=pd.DataFrame(selection.get('candidate_assessments',[selection]))
    assessments.drop(columns=['paired_subject_bootstrap_ci95'],errors='ignore').to_csv(run/'promotion_assessments.csv',index=False)
    model=Path(final['path']);parity=rt.read_json(model/'onnx_real_parity.json')
    smoke=rt.read_json(run/'inference_smoke.json')
    conclusions=[]
    for row in first.to_dict('records'):
        name=row['experiment'].rsplit('_seed',1)[0]
        if name=='control':reason='Matched control reproduced the original frozen split exactly; retained as reference.'
        elif name in set(ranking.candidate):reason='Confirmed over seeds 302-304; final decision also uses matched subject-wise CV and non-degradation gates.'
        else:reason='Not shortlisted: lower clean subject Macro-F1 at screen seed 302. No further search or stacking.'
        conclusions.append(f"* {name}: {row['subject_macro_f1']:.4f}; {reason}")
    original=rt.read_json(run/'experiments/control_seed302/result.json')['metrics']
    ci=selection['paired_subject_bootstrap_ci95']
    counts=Counter()
    for path in (run/'experiments').glob('*/result.json'):
        name=path.parent.name
        counts['smoke' if name.startswith('smoke') else 'cv' if name.startswith('cv') else 'final_fit' if name=='final_development_fit' else 'matched_split']+=1
    final_history=pd.read_csv(run/'experiments/final_development_fit/history.csv')
    chosen=selection['candidate']
    chosen_rows=df[df.experiment.str.startswith(chosen+'_seed')]
    command=' '.join(smoke['command'])
    text=f'''# PoseForge model improvement before external testing

## Outcome and evaluation boundary

Recommended model: `{final['recommended_model']}`.
New candidate promoted: **{selection['promoted']}**.
Original `artifacts/models/mlp_wide/v001` is unchanged. No QBAO prediction,
test-metric read for selection, new final test, or external test was performed.
The physical development snapshot contains only the twelve allowed subjects.

Original matched-split baseline: seed-302 mean subject Macro-F1 **{original['subject_macro_f1']:.4f}**;
three-seed mean **{float(ranking[ranking.candidate=='control'].mean_f1.iloc[0]):.4f}**.
Best confirmed non-control candidate: **{chosen}**.
Matched mean gain: **{selection['matched_validation_gain']:+.4f}**.
Matched development CV gain: **{selection['cv_gain']:+.4f}**.
These are distinct protocols; their absolute means are not directly compared.

## Dataset quality and curation

Development only: {quality['subjects']} subjects, {quality['source_takes']} source takes,
{quality['clips']} camera clips, {quality['frames']} frames, **{quality['eligible']} eligible**,
and {quality['excluded']} excluded observations. All original eligibility masks
were preserved exactly, keeping the historical validation comparison comparable.
The new derived version isolates development data; no curation decision was based
on old held-out accuracy. Source/processed hashes, source indices and timestamps
are retained. No recovery, relabeling, coordinate filling, video modification,
keypoint smoothing, or YOLO retraining occurred.

Fifteen of the previously sixteen pose-empty camera clips belong to development.
They still contain no selected pose and require review; the other clip belongs
to the excluded subject. Selected multiperson observations retain the conservative
geometric duplicate-box check, not an assumption based on ByteTrack ID. Exclusion
records state no selected pose, uncertain selected person, fewer than four measured
joints, or an existing curation/camera exclusion. Clips are not dropped merely
for partial joints. Per-frame reasons and review flags are in
`exclusions_and_review.csv` and `quality_frames.csv.gz`.

Quality distributions by label, camera, subject and their intersections are in
`quality_*.csv`. Front camera: 6,433/7,647 eligible; oblique: 6,310/6,479.
This camera imbalance reflects 1,214 versus 169 excluded observations. Development
capture spans average {quality['mean_clip_duration_s']:.3f} s; the recordings are
short and neighboring frames are strongly correlated. There are
{quality['timestamp_duplicates']} repeated timestamp ticks and {quality['large_timestamp_gaps']}
gaps above a fixed 250 ms threshold. This gap definition differs from the prior
adaptive audit threshold. Source frame order was never changed. Subject-wise folds
and subject-level uncertainty avoid treating correlated frames as independent people.

## Development error analysis

The matched control's 144 errors consist of 98 VMTRI hunched-to-desk predictions,
42 VMTRI backward-to-upright predictions, and four TRUONG desk-to-backward
predictions. The dominant clips contain valid nose observations, so missing head
joints alone do not explain the errors. Existing diagnostics show overlapping
normalized head/torso geometry for these held postures across subjects.
Geometry features and normalization/residual architectures were prioritized for
that reason. No source labels were changed based on predictions.

Control front Macro-F1: {original['per_camera'][0]['macro_f1']:.4f};
oblique: {original['per_camera'][1]['macro_f1']:.4f}.
Labels 4 and 5 had no control-validation errors, but partial arm movement is not
a separately labeled condition. Low-confidence predictions (<0.70), visibility
bins, per-subject/camera error pairs and synthetic missing-joint sensitivity are
saved in `validation_error_records.csv.gz`, `low_confidence_analysis.csv`,
`missing_joint_sensitivity.csv`, and each experiment's `robustness_latency.json`.
Synthetic stress removes left/right elbow+wrist or nose while preserving masks;
it is a development robustness diagnostic, not an external test.

## Feature and architecture ablation

The original `pose13_xy_conf_mask_causal_v2` and original Small/Base/Wide models
remain fully supported. The separate experimental contract is
`{GEOMETRY_VERSION}`: 52 original values + 22 geometry values + 22 validity masks,
**96 dimensions**. It uses torso/head displacement and inclination in 2D,
shoulder/hip alignment, elbow cosines, wrist/shoulder height and torso asymmetry.
Each calculation requires its measured joints; degenerate denominators invalidate
the feature; calculated values are bounded to [-4,4]. No depth is invented.
The same NumPy/Torch-dispatched function is used by training and deployment,
with parity/missing-joint tests. Causal base normalization is unchanged.

Residual MLP uses a 128-wide stem and two 128-wide skip blocks. LayerNorm MLP
adds LayerNorm to the original 256/128 Wide dimensions. Quality-gated augmentation
only affects poses with at least ten measured joints and mean measured confidence
>=0.70. It applies small jitter, confidence degradation and at most one forearm
joint occlusion, preserving head/torso anchors. Low-quality rows are untouched.
No repeated class-weighting/smoothing sweep, temporal model, or blind combination
of changes was introduced.

Screen seed 302; the same frozen 10-train/2-validation subjects and 1,433 eligible
validation observations for every row:

{markdown_table(first,['experiment','input_dim','subject_macro_f1','frame_macro_f1','clip_macro_f1','calibration_ece','best_epoch','epochs'])}

{chr(10).join(conclusions)}

## Multi-seed confirmation and grouped CV

Matched split, seeds 302/303/304 (sample standard deviations across three seeds):

{markdown_table(ranking,['candidate','mean_f1','std_f1','mean_clip_f1','median_epoch','arm_occlusion_f1'])}

Three seeded subject folds contain four validation and eight training subjects
each. Every take, camera, session and derived frame of a subject stays in one fold.
The folds and promotion thresholds were recorded before screening in `protocol.json`.
CV initially compares control and the strongest confirmed challenger, at seeds 302 and 303.
Epoch counts are fixed from median best epochs in the matched-split confirmation;
fold validation never selects its checkpoint. CV uses development subjects that
also inform screening and schedule selection, so it is development selection
evidence, not an unbiased final generalization estimate.

After geometry failed CV, the already-confirmed targeted-arm augmentation was
evaluated under the same folds, seeds, fixed schedule and unchanged promotion gates.
Its +0.1189 matched arm-occlusion gain with no clean matched degradation justified
this bounded robustness follow-up; `cv_robustness_plan.json` records that decision
before its fold training. The original frozen `cv_plan.json` was not overwritten.

{markdown_table(cv_summary,['candidate','subject_macro_f1','clip_macro_f1','arm_occlusion_f1'])}

Candidate-specific decisions:

{markdown_table(assessments,['candidate','matched_validation_gain','cv_gain','cv_arm_occlusion_gain','positive_folds','min_subject_delta','min_class_recall_delta','promoted'])}

Positive fold means for the challenger: **{selection['positive_folds']}/3**.
Arm-occlusion CV gain: **{selection['cv_arm_occlusion_gain']:+.4f}**.
Paired subject Macro-F1 gain bootstrap 95% interval: **[{ci[0]:+.4f}, {ci[1]:+.4f}]**,
using 20,000 resamples of twelve subjects after averaging each subject over two seeds.
The interval is descriptive after development selection and does not correct for
multiple candidate selection. Statistically convincing under this interval:
**{selection['statistically_convincing']}**. Practically convincing under the
pre-recorded promotion rules: **{selection['practically_convincing']}**.
TDUY lacks eligible class 5; per-subject scores retain the same fixed six-class
definition for both models rather than silently dropping its missing class.

Promotion required >=0.01 matched and CV subject-F1 gains with at least two positive
folds, or >=0.02 arm-occlusion gain with at most 0.005 clean degradation. Both routes
also required no material camera/class/worst-subject regressions. Actual worst
deltas: subject {selection['min_subject_delta']:+.4f}, camera
{selection['min_camera_delta']:+.4f}, class recall {selection['min_class_recall_delta']:+.4f}.
Clean-gain gate: {selection['clean_gain_gate']}; robustness gate:
{selection['robustness_gate']}; non-degradation gate: {selection['non_degradation_gate']}.
No thresholds or candidate settings were changed after these results.

Complete fold/seed and per-subject rows are in `cross_validation.csv` and
`cv_subject_metrics.csv`. Every candidate includes per-class precision/recall/F1,
confusion matrix, camera metrics, frame/clip/source metrics, histories, learning
curves, best checkpoints, and resumable optimizer/RNG checkpoints.

## Model size, latency, final development fit and deployment

{markdown_table(latency,['candidate','input_dim','parameters','cpu_batch1_median_ms','cpu_batch1_p95_ms'])}

CPU figures use one Torch thread, 100 warmups and 1,000 measured batch-one calls,
excluding feature construction and pose extraction. GPU wall-time measurements
are also saved per experiment; hardware clocks and launch overhead make tiny-model
GPU times noisy. End-to-end ONNX CPU predict-call timings including geometry and
softmax are in `deployment_latency.csv`.

Actual training: CUDA on NVIDIA GeForce RTX 2050; PyTorch 2.6.0+cu124. AdamW,
lr=0.001, weight_decay=0.001, batch=128, dropout=0.15, float32, max_epochs=100,
patience=12 for matched-split runs. No working packages were reinstalled. Completed
runs: {dict(counts)}. A five-epoch real CUDA smoke verified loss reduction first.

A fresh development-only final fit used all twelve allowed subjects / 12,743
eligible observations, seed **10302**, and **{selection['final_epochs']} epochs**.
Loss changed from {final_history.train_loss.iloc[0]:.4f} to
{final_history.train_loss.iloc[-1]:.4f}. Checkpoint:
`{model/'model_final.pt'}`. Status: {'promoted' if final['promoted'] else 'unpromoted research reproduction; recommended release remains v001'}.
No new test metrics file was created. External evaluation remains pending.

ONNX parity on all 12,743 development features: verified={parity['verified']}; maximum
absolute probability error **{parity['max_abs_probability_difference']:.9g}** (limit 1e-5).
Both PT/ONNX loaders produce six probabilities, and original v001 still loads
and matches its ONNX implementation. Original 52-dimensional normalization and
all baseline artifact hashes remain intact. Preservation: {preservation}.

Development-footage smoke: **{smoke['frames']} frames**, {smoke['classified']} classified;
YOLO11n-Pose -> ByteTrack -> existing selector/causal normalizer -> classifier.
This is not a final external evaluation. Its local annotated video is excluded
from shareable artifacts. Executed command:

```powershell
{command}
```

## Files, reproducibility, limitations and readiness

Created `shared/pose_geometry.py`, `shared/model_improvement.py`,
`scripts/improve_poseforge.py`, and `tests/test_model_improvement.py`.
Extended `shared/mlp_model.py` with a separate experimental architecture factory;
extended reusable `shared/real_training.py` through optional data transforms/fixed
schedules; extended `scripts/export_onnx.py` and `inference/pipeline.py` with explicit
feature-contract/architecture dispatch. Existing base contracts/model definitions
were preserved. `docs/TRAINING_WINDOWS.md` records commands actually executed.

All current reports/artifacts are under `{run}`. Candidate JSON configurations,
immutable data hashes, frozen folds/promotion rules, logs, and optimizer/RNG states
support reproduction and safe interruption via `--resume`. New final fits are
versioned and never overwrite a prior model. No external test stage exists in the
new CLI. Raw recordings, recovered keypoints and recovery backups were not modified.
Relevant tests and their actual output are saved in `regression_after.log`.
The full regression suite finished with **119 passed, 4 skipped, 1 failed**.
The remaining pre-existing failure is the asset-migration symbolic-link test:
Windows denies link creation (WinError 1314). Collector GUI tests passed.
The focused feature/loader/resume regression suite finished with **37 passed,
2 skipped**, including experimental geometry ONNX round trips for residual and
LayerNorm architectures. See `regression_feature_contracts.log`.

Remaining weaknesses: twelve development subjects, short highly correlated takes,
15 pose-empty development clips, much poorer front-camera coverage, subject-specific
hunch/desk and upright/backward ambiguity, and no independent labels for gradual
arm movement. A bootstrap over twelve subjects cannot establish broad population
accuracy. No VLM relabeling or temporal model was used.

Ready for a newly authorized independent external test: **yes, using the recommended
release above**, with preprocessing frozen and these limitations recorded. This
phase stops at development training/deployment verification; external testing must
be explicitly authorized later.
'''
    path=root/'artifacts/reports/MODEL_IMPROVEMENT_REPORT.md'
    if path.exists():raise FileExistsError('Preserve existing improvement report; use a versioned report name')
    path.write_text(text,encoding='utf-8')
    (run/'MODEL_IMPROVEMENT_REPORT.md').write_text(text,encoding='utf-8')
    # Explicit source copies, never source videos or feature arrays, for review.
    source_dir=run/'source_code';source_dir.mkdir(exist_ok=True)
    names=['shared/pose_geometry.py','shared/model_improvement.py','shared/mlp_model.py','shared/real_training.py',
           'shared/pose_features.py','inference/pipeline.py','inference/run.py','scripts/export_onnx.py','scripts/improve_poseforge.py']
    for name in names:
        target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    rt.write_json(run/'code_hashes.json',{name:file_sha256(root/name) for name in names})
    rt.write_json(run/'completion_summary.json',dict(selection=selection,final=final,onnx=parity,preservation=preservation,
                  report=str(path),external_test_executed=False,completed_runs=dict(counts)))
    return str(path)
