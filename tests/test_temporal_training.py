"""Temporal time/provenance/embargo rules without modifying any real recording."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from shared import temporal_training as tt
from shared import real_training as rt
from shared.dataset_pipeline import file_sha256


def test_boundaries_and_symmetric_purge_use_capture_nanoseconds():
    assert tt.temporal_boundaries(0,10_000_000_000)==(8_000_000_000,9_000_000_000)
    ticks=np.array([0,7_749_999_999,7_750_000_000,8_000_000_000,8_250_000_000,8_250_000_001,8_749_999_999,8_750_000_000,9_250_000_001])
    planned,purged=tt.assign_capture_times(ticks,0,10_000_000_000)
    assert purged.tolist()==[False,False,True,True,True,False,False,True,False]
    assert planned[5]=='val' and planned[-1]=='internal_test'
    with pytest.raises(ValueError,match='sum'):tt.temporal_boundaries(0,100,(.8,.2,.1))
    with pytest.raises(ValueError,match='bounds'):tt.temporal_boundaries(10,1)


def timestamps(ticks,utc_offset_ns=1_700_000_000_000_000_000):
    ticks=np.asarray(ticks,np.int64)
    return pd.DataFrame(dict(writer_frame_index=range(len(ticks)),source_frame_index=range(100,100+len(ticks)),
                             host_recv_monotonic_ns=ticks,host_recv_utc=pd.to_datetime(ticks+utc_offset_ns,utc=True).astype(str)))


def test_paired_camera_common_clock_accepts_duplicates_and_rejects_offset():
    a=timestamps([1_000_000_000,1_100_000_000,1_100_000_000,2_000_000_000])
    b=timestamps([1_020_000_000,1_120_000_000,1_920_000_000])
    metadata=dict(start_host_monotonic_ns=1_000_000_000,end_host_monotonic_ns=2_000_000_000)
    assert tt.verify_camera_clocks({'cam01':a,'cam02':b},metadata)['verified']
    shifted=timestamps(b.host_recv_monotonic_ns,utc_offset_ns=1_700_000_001_000_000_000)
    assert not tt.verify_camera_clocks({'cam01':a,'cam02':shifted},metadata)['verified']
    backward=a.iloc[[0,3,1,2]].reset_index(drop=True)
    assert not tt.verify_camera_clocks({'cam01':backward,'cam02':b},metadata)['verified']


def frame_table(duration=10.):
    records=[]
    for camera,offset in [('cam01',0),('cam02',10_000_000)]:
        ticks=np.arange(offset,int(duration*1e9),25_000_000,dtype=np.int64)
        planned,purge=tt.assign_capture_times(ticks,0,int(duration*1e9),purge_seconds=.25)
        for i,tick in enumerate(ticks):
            records.append(dict(person_id='P',session_id='S',source_take_id='take_001',source_group_id='P/S/0/take_001',
                       camera_id=camera,source_frame_index=i,clip_id='P_'+camera,capture_ns=int(tick),label=0,
                       eligible=True,embargo=bool(purge[i]),clock_verified=True,planned_partition=planned[i]))
    return pd.DataFrame(records)


def test_camera_views_use_same_boundaries_despite_different_frame_indices():
    original=frame_table()
    frames,segments=tt.apply_segment_quality(original)
    b={'P/S/0/take_001':dict(start_ns=0,end_ns=10_000_000_000)}
    assert tt.leakage_checks(frames,b,(.8,.1,.1),.25)['passed']
    for part in tt.PARTITIONS:
        assert frames.loc[frames.partition==part,'camera_id'].nunique()==2
    # Alter one camera's temporal assignment: shared-clock check must reject it.
    bad=frames.copy();idx=bad.index[bad.partition.eq('val')][0];bad.loc[idx,'partition']='train'
    with pytest.raises(ValueError,match='Cross-camera'):tt.leakage_checks(bad,b,(.8,.1,.1),.25)


def test_short_video_rejects_validation_without_reassigning_to_train():
    q=frame_table(2.4)
    result,segments=tt.apply_segment_quality(q)
    assert not result.partition.eq('val').any()
    assert (result.loc[q.planned_partition.eq('val'),'partition']=='excluded').all()
    assert not segments.loc[segments.partition.eq('val'),'usable'].any()


def test_duplicate_identity_and_excluded_frame_assignment_are_rejected():
    frames,_=tt.apply_segment_quality(frame_table())
    b={'P/S/0/take_001':dict(start_ns=0,end_ns=10_000_000_000)}
    with pytest.raises(ValueError,match='Duplicate'):
        tt.leakage_checks(pd.concat([frames,frames.iloc[:1]],ignore_index=True),b,(.8,.1,.1),.25)
    frames.loc[frames.partition.eq('train').idxmax(),'eligible']=False
    with pytest.raises(ValueError,match='Quality-excluded'):tt.leakage_checks(frames,b,(.8,.1,.1),.25)


def prepared_fixture(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'processed').mkdir()
    catalog=[];frames=[]
    for person,label in [('A',0),('QBAO',1)]:
        clip=person+'_cam01'
        x=np.ones((4,13,4),np.float32);x[:,:,:2]=label+.25
        eligible=np.array([True,True,True,False])
        np.savez_compressed(run/'processed'/f'{clip}.npz',x=x,eligible=eligible)
        catalog.append(dict(clip_id=clip,label=label,person_id=person,camera_id='cam01',camera_view='front',source_group_id=person))
        for i,part in enumerate(['train','val','internal_test','excluded']):
            frames.append(dict(clip_id=clip,frame_index=i,person_id=person,eligible=bool(eligible[i]),partition=part,capture_ns=i))
    pd.DataFrame(catalog).to_csv(run/'catalog_source.csv',index=False)
    pd.DataFrame(frames).to_csv(run/'temporal_split_manifest.csv',index=False)
    rt.write_json(run/'frozen_checksums.json',{})
    rt.seal(run/'processed')
    import shutil
    (run/'full_snapshot').mkdir()
    shutil.copytree(run/'processed',run/'full_snapshot/processed')
    return run


def test_full_data_uses_all_eligible_including_qbao_and_internal_partitions(tmp_path):
    run=prepared_fixture(tmp_path)
    temporal=tt.FramePartitionData(run,['train','val'],'cpu')
    assert len(temporal.x)==4
    assert all(r['partition']!='internal_test' for r in temporal.frame_rows)
    full=tt.FramePartitionData(run,['train'],'cpu',full=True)
    assert len(full.x)==6 and {c['person'] for c in full.clips}=={'A','QBAO'}
    assert any(r['partition']=='internal_test' for r in full.frame_rows)
    assert all(r['eligible'] for r in full.frame_rows)


def test_previous_model_preservation_detects_changes(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();file=raw/'pose.txt';file.write_text('original')
    old=tmp_path/'v001.pt';old.write_bytes(b'checkpoint')
    run=tmp_path/'run';(run/'audit').mkdir(parents=True)
    rt.write_json(run/'run_config.json',dict(raw_root=str(raw)))
    rt.write_json(run/'audit/source_inventory.json',{'pose.txt':{'sha256':file_sha256(file)}})
    rt.write_json(run/'prior_models_hashes.json',{str(old):file_sha256(old)})
    assert tt.verify_sources(run)['prior_model_files_unchanged']==1
    old.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Earlier model'):tt.verify_sources(run)


def test_temporal_training_adapter_runs_without_internal_test_observations(tmp_path):
    run=prepared_fixture(tmp_path);torch.set_num_threads(2)
    cfg={**tt.baseline_config(302),'epochs':2,'patience':2,'batch_size':2}
    data=tt.FramePartitionData(run,['train','val'],'cpu')
    result=rt.run_candidate(run,cfg,'cpu','synthetic_adapter_contract',data=data)
    assert result['metrics']['frames']==2
    assert len(data.ids([c for c in data.clips if c['split']=='train']))==2
