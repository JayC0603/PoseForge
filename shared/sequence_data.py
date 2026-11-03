"""One timing/window contract for offline datasets and per-stream online buffers."""
from collections import deque
import math
import numpy as np


def sequence_contract(length=16,stride=4,min_coverage=.8,max_gap_seconds=.25,max_missing_run=3):
    if length not in (16,32) or stride<1 or not 0<min_coverage<=1 or max_gap_seconds<=0 or max_missing_run<0:
        raise ValueError('Invalid sequence/timing configuration')
    return dict(version='poseforge_sequence_v1',sequence_length=int(length),input_dim=52,hidden_dim=64,layers=1,
                bidirectional=False,training_stride=int(stride),evaluation_stride=1,min_usable_coverage=float(min_coverage),
                max_timestamp_gap_seconds=float(max_gap_seconds),max_consecutive_missing=int(max_missing_run),
                duplicate_ticks='retained in original order; recorded in lineage',non_monotonic='reset/reject spanning window',
                padding_policy='reject; no fabricated frames',target_policy='current frame must be eligible',
                tracking='reset on changed selected track or track session; IDs are temporary stream state only',
                warmup='abstain until a full valid window; never reuse stale probabilities',
                feature_contract='pose13_xy_conf_mask_causal_v2')


def reset_edges(ticks,eligible,tracks=None,sessions=None,contract=None):
    contract=contract or sequence_contract()
    ticks=np.asarray(ticks,np.int64);eligible=np.asarray(eligible,bool)
    edges=np.zeros(len(ticks),bool);reasons={};missing=0;last_track=None;last_session=None
    for i in range(len(ticks)):
        reason=[]
        if i and ticks[i]<ticks[i-1]:reason.append('backward_timestamp')
        if i and ticks[i]-ticks[i-1]>contract['max_timestamp_gap_seconds']*1e9:reason.append('large_timestamp_gap')
        session=None if sessions is None else sessions[i]
        track=None if tracks is None else tracks[i]
        if session is not None and last_session is not None and session!=last_session:reason.append('track_session_change')
        if track is not None and last_track is not None and track!=last_track:reason.append('selected_track_change')
        missing=0 if eligible[i] else missing+1
        if missing>contract['max_consecutive_missing']:reason.append('long_missing_run')
        if reason:edges[i]=True;reasons[i]=reason
        if track is not None:last_track=track
        if session is not None:last_session=session
    return edges,reasons


def window_records(ticks,eligible,tracks,sessions,contract,stride=1):
    ticks=np.asarray(ticks,np.int64);eligible=np.asarray(eligible,bool)
    length=contract['sequence_length'];edges,reasons=reset_edges(ticks,eligible,tracks,sessions,contract)
    result=[]
    for end in range(0,len(ticks),stride):
        start=end-length+1;reason=[]
        if start<0:reason.append('warmup_short_context')
        if not eligible[end]:reason.append('ineligible_target')
        if start>=0:
            if edges[start+1:end+1].any():reason.append('reset_boundary_in_window')
            if eligible[start:end+1].mean()<contract['min_usable_coverage']:reason.append('insufficient_usable_coverage')
        result.append(dict(start_frame=max(0,start),end_frame_exclusive=end+1,target_frame=end,
                           start_capture_ns=int(ticks[max(0,start)]),end_capture_ns=int(ticks[end]),
                           usable_frames=int(eligible[max(0,start):end+1].sum()),
                           duplicate_ticks=int((np.diff(ticks[max(0,start):end+1])==0).sum()),
                           accepted=not reason,rejection_reason=';'.join(reason)))
    return result,reasons


class TemporalSequenceBuffer:
    """Owned by one PosePipeline, never by the shared classifier/model."""
    def __init__(self,contract):
        self.contract=contract;self.reset()

    def reset(self):
        self.features=deque(maxlen=self.contract['sequence_length'])
        self.usable=deque(maxlen=self.contract['sequence_length'])
        self.last_ns=None;self.last_track=None;self.missing=0

    def push(self,features,timestamp_ns,usable,track_id=None):
        timestamp_ns=int(timestamp_ns)
        reset=(self.last_ns is not None and (timestamp_ns<self.last_ns or timestamp_ns-self.last_ns>self.contract['max_timestamp_gap_seconds']*1e9))
        reset|=track_id is not None and self.last_track is not None and track_id!=self.last_track
        if reset:self.reset()
        self.missing=0 if usable else self.missing+1
        if self.missing>self.contract['max_consecutive_missing']:
            self.reset();self.missing=self.contract['max_consecutive_missing']+1
        self.last_ns=timestamp_ns
        if track_id is not None:self.last_track=track_id
        self.features.append(np.asarray(features,np.float32).reshape(52).copy());self.usable.append(bool(usable))
        if not usable or len(self.features)<self.contract['sequence_length'] or np.mean(self.usable)<self.contract['min_usable_coverage']:
            return None
        return np.stack(self.features)
