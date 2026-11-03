"""Take grouping, causal window contracts, coverage fairness and GRU deployment."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from shared import real_training as rt
from shared.take_gru_training import allocate_takes,check_leakage,TakeData
from shared.sequence_data import sequence_contract,window_records,TemporalSequenceBuffer
from shared.mlp_model import build_classifier_model,PostureMLP


def groups():
    return pd.DataFrame([dict(source_group_id=f'P{p}/S/label_{label}/take_001',person_id=f'P{p}',label=label,
                              eligible_frames=80+p,valid32_targets=20) for p in range(5) for label in range(6)])


def test_deterministic_stratified_take_allocation_and_camera_pairing():
    a,info,_=allocate_takes(groups(),302,64);b,_,_=allocate_takes(groups(),302,64)
    assert a==b and not info['model_scores_used']
    catalog=pd.DataFrame([dict(clip_id=f'{r.source_group_id}_{cam}',source_group_id=r.source_group_id,
                     person_id=r.person_id,label=r.label,usable_frames=40,camera_id=cam,partition=a[r.source_group_id])
                     for r in groups().itertuples() for cam in ['cam01','cam02']])
    assert check_leakage(catalog)['passed']
    assert catalog.groupby('source_group_id').partition.nunique().eq(1).all()
    bad=catalog.copy();bad.loc[0,'partition']='test' if bad.loc[0,'partition']!='test' else 'train'
    with pytest.raises(ValueError,match='spans'):check_leakage(bad)


@pytest.mark.parametrize('length',[16,32])
def test_window_order_shape_missing_timing_and_prefix_causality(length):
    n=80;ticks=np.arange(n,dtype=np.int64)*33_000_000;eligible=np.ones(n,bool)
    ticks[8]=ticks[7] # repeated ticks retained, not silently dropped
    eligible[10:12]=False
    ticks[50:]+=1_000_000_000
    tracks=[1]*n;sessions=['S']*n;contract=sequence_contract(length)
    records,resets=window_records(ticks,eligible,tracks,sessions,contract)
    assert 50 in resets
    assert all(not r['accepted'] for r in records[:length-1])
    assert all(not r['accepted'] for r in records[50:min(n,50+length-1)])
    assert any(r['duplicate_ticks'] for r in records)
    prefix,_=window_records(ticks[:40],eligible[:40],tracks[:40],sessions[:40],contract)
    assert records[:40]==prefix
    for r in records:
        if r['accepted']:assert r['end_frame_exclusive']-r['start_frame']==length and eligible[r['target_frame']]


def test_long_missing_run_and_track_changes_reset_context():
    n=70;ticks=np.arange(n)*33_000_000;usable=np.ones(n,bool);usable[20:25]=False
    tracks=[1]*40+[2]*30
    records,resets=window_records(ticks,usable,tracks,['S']*n,sequence_contract())
    assert 23 in resets and 40 in resets
    assert all(not r['accepted'] for r in records[40:55])
    assert records[55]['accepted']


def test_online_buffer_matches_offline_targets_and_is_per_stream():
    n=80;ticks=np.arange(n,dtype=np.int64)*33_000_000;usable=np.ones(n,bool);usable[20:22]=False
    tracks=[1]*40+[2]*40;contract=sequence_contract(16)
    records,_=window_records(ticks,usable,tracks,['S']*n,contract)
    online=TemporalSequenceBuffer(contract);other=TemporalSequenceBuffer(contract);accepted=[]
    for i in range(n):
        x=np.full(52,i,np.float32) if usable[i] else np.zeros(52,np.float32)
        window=online.push(x,ticks[i],usable[i],tracks[i])
        if window is not None:
            accepted.append(i);assert window.shape==(16,52) and window[-1,0]==i
    assert accepted==[r['target_frame'] for r in records if r['accepted']]
    assert other.push(np.ones(52,np.float32),1,True,1) is None
    online.reset();assert online.push(np.ones(52,np.float32),9999999999,True,2) is None


def fixture_run(tmp_path):
    run=tmp_path/'run';(run/'processed').mkdir(parents=True);(run/'sequences').mkdir()
    rows=[];window_sets={16:[],32:[]};common=[]
    for part in ['train','val','test']:
        for label in range(6):
            for camera in ['cam01','cam02']:
                group=f'P/S/{label}/{part}';clip=f'{part}_{label}_{camera}';n=40
                x=np.ones((n,13,4),np.float32);x[:,:,0]=np.arange(n)[:,None]/100
                eligible=np.ones(n,bool);ticks=np.arange(n,dtype=np.int64)*33_000_000
                np.savez_compressed(run/'processed'/f'{clip}.npz',x=x,eligible=eligible,capture_ns=ticks)
                rows.append(dict(clip_id=clip,source_group_id=group,person_id='P',camera_id=camera,camera_view=camera,
                                 label=label,usable_frames=n,partition=part))
                for length in [16,32]:
                    records,_=window_records(ticks,eligible,[1]*n,['S']*n,sequence_contract(length))
                    for record in records:
                        record.update(clip_id=clip,partition=part,training_selected=(record['target_frame']-length+1)%4==0)
                        window_sets[length].append(record)
                common.extend(dict(clip_id=clip,target_frame=i,partition=part) for i in range(31,n))
    pd.DataFrame(rows).to_csv(run/'split_manifest.csv',index=False)
    rt.write_json(run/'frozen_checksums.json',{})
    for length,records in window_sets.items():pd.DataFrame(records).to_csv(run/f'sequences/windows{length}.csv.gz',index=False,compression='gzip')
    pd.DataFrame(common).to_csv(run/'sequences/common_targets.csv',index=False)
    rt.seal(run/'processed');rt.seal(run/'sequences')
    return run


def test_no_window_crosses_take_camera_or_partition_and_coverage_is_matched(tmp_path):
    run=fixture_run(tmp_path);targets=[]
    for kind in ['mlp_control','gru16','gru32']:
        data=TakeData(run,kind,['train','val'],'cpu')
        assert not any(r['partition']=='test' for r in data.metadata)
        ids=data.ids([c for c in data.clips if c['split']=='val'])
        targets.append(data.target_keys[ids].tolist())
        if kind.startswith('gru'):
            length=int(kind[3:]);assert data.x.shape[1:]==(length,52)
            for i in ids:assert np.isclose(data.x[i,-1,0],data.metadata[i]['target_frame']/100)
    assert targets[0]==targets[1]==targets[2]


@pytest.mark.parametrize('name',['gru16','gru32'])
def test_actual_gru_forward_backward_and_onnx_roundtrip(tmp_path,name):
    from shared.dataset_pipeline import COCO13
    from inference.pipeline import PostureClassifier
    from scripts.export_onnx import export
    length=int(name[3:]);model=build_classifier_model(name)
    assert isinstance(build_classifier_model('mlp_wide'),PostureMLP)
    x=torch.randn(3,length,52);loss=torch.nn.CrossEntropyLoss()(model(x),torch.tensor([0,1,2]));loss.backward()
    assert torch.isfinite(loss) and model.gru.bidirectional is False
    contract=dict(version='pose13_xy_conf_mask_causal_v2',flatten_dim=52,shape_per_frame=[13,4],
         channels=['x_norm','y_norm','confidence','valid_mask'],joints=COCO13,confidence_threshold=.35,ema_alpha=.25)
    rt.write_json(tmp_path/'feature_contract.json',contract);rt.write_json(tmp_path/'sequence_contract.json',sequence_contract(length))
    torch.save(dict(stage='final',model_name=name,model_config={'dropout':.15},feature_contract=contract,
                    label_names=[str(i) for i in range(6)],model_state_dict=model.state_dict()),tmp_path/'model_final.pt')
    export(tmp_path)
    pt=PostureClassifier(tmp_path,'pt');onnx=PostureClassifier(tmp_path,'onnx')
    sample=x.detach().numpy()[0]
    np.testing.assert_allclose(pt.predict(sample),onnx.predict(sample),atol=1e-5,rtol=1e-5)
    assert pt.predict(sample).shape==(6,)


def test_pipeline_causal_warmup_and_track_reset_without_stale_predictions(monkeypatch):
    from inference import pipeline
    class Replay:
        def __init__(self,*args,**kwargs):self.i=0
        def track_frame(self,frame):
            i=self.i;self.i+=1
            if 18<=i<23:return []
            kp=[[float(50+j),float(30+2*j),.9] for j in range(17)]
            return [dict(bbox=[0,0,200,200],keypoints=kp,track_id=1 if i<40 else 2)]
        def reset_tracking(self):pass
        def close(self):pass
    class Classifier:
        conf_thresh=.35;ema_alpha=.25;is_temporal=True;sequence_contract=sequence_contract(16)
        labels=[str(i) for i in range(6)]
        def predict(self,x):
            assert x.shape==(16,52)
            return np.array([.8,.04,.04,.04,.04,.04])
    monkeypatch.setattr(pipeline,'PoseRunner',Replay)
    stream=pipeline.PosePipeline('unused',Classifier())
    rows=[stream.process(np.zeros((220,220,3),np.uint8),timestamp_ns=i*33_000_000)[0] for i in range(65)]
    assert all(r['class_id'] is None for r in rows[:15])
    assert rows[15]['class_id']==0
    assert all(r['class_id'] is None for r in rows[18:23])
    assert all(r['class_id'] is None for r in rows[40:55])
    assert rows[55]['class_id']==0
    stream.reset();assert len(stream.temporal.features)==0


def test_external_guard_requires_new_consented_isolated_data(tmp_path):
    from scripts.evaluate_real_world import validate_external
    training=tmp_path/'training';training.mkdir();external=tmp_path/'external';external.mkdir()
    video=external/'new.mp4';video.write_bytes(b'new-video')
    (external/'ticks.csv').write_text('writer_frame_index,host_recv_monotonic_ns\n0,1\n')
    row=dict(person_id='NEW01',session_id='S',take_id='T',camera_id='cam01',label=0,video_path='new.mp4',timestamps_path='ticks.csv')
    manifest=dict(dataset_role='independent_external_test',consent_confirmed=True,recordings=[row])
    assert len(validate_external(external,manifest,training,{},['OLD']))==1
    with pytest.raises(ValueError,match='consent'):
        validate_external(external,{**manifest,'consent_confirmed':False},training,{},['OLD'])
    with pytest.raises(ValueError,match='new volunteers'):
        validate_external(external,manifest,training,{},['NEW01'])
    from shared.dataset_pipeline import file_sha256
    with pytest.raises(ValueError,match='Copied'):
        validate_external(external,manifest,training,{'old.mp4':{'sha256':file_sha256(video)}},['OLD'])
