"""No physical camera required. Exercise real Qt, spawn lifecycle and causal state."""
import json
import os
import queue
import time

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from inference import pipeline
from inference.gui import MainWindow, SourcePanel, VideoPreview, frame_rect, map_to_frame
from inference.models import ModelEntry, DEFAULT_MODEL, validate_bundle
from inference.runtime import DesktopReader, SessionProcess, publish
from inference.telemetry import TimingWindow, DisplayRate


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, seconds=8):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        app.processEvents()
        if predicate():return
        QTest.qWait(10)
    assert predicate(), 'Condition did not complete before timeout'


def fake_worker(config,stop,outbox,controls):
    """Spawn-safe synthetic source, no Qt/camera/model in child."""
    frame=np.zeros((120,160,3),np.uint8)
    i=0
    while not stop.wait(.02):
        i+=1
        publish(outbox,dict(state='RUNNING',info=dict(class_id=0,label='A',score=.9,
            probs=[.9,.02,.02,.02,.02,.02],keypoints_valid=13,track_id=i,selection_reason='single_detection'),
            frame=frame,received=time.monotonic(),packet_no=i,telemetry={'processed_frames':i,'yolo_fps':10.,'classifier_fps':1000.,'pipeline_fps':9.}))
    publish(outbox,dict(state='STOPPED',info={}))


def blocked_worker(config,stop,outbox,controls):
    # Simulate an uninterruptible native camera read. Process containment must
    # stop this without ever blocking QApplication or leaking workers.
    time.sleep(30)


class FakeSession:
    def __init__(self,config):
        self.config=config;self.finished=False;self.forced=False;self.active=False;self.roi=None;self.calls=0
    def start(self):self.active=True
    def stop(self):self.active=False;self.finished=True
    def alive(self):return self.active
    def poll(self):
        if not self.active:return dict(state='STOPPED',info={})
        self.calls+=1
        return dict(state='RUNNING',info=dict(class_id=0,label='A',score=.9,probs=[.9,.02,.02,.02,.02,.02],
            keypoints_valid=13,track_id=1,selection_reason='single_detection'),
            frame=np.zeros((120,160,3),np.uint8),received=time.monotonic(),
            telemetry=dict(processed_frames=self.calls,yolo_fps=20.,classifier_fps=2000.,pipeline_fps=18.))
    def set_roi(self,value):self.roi=value


@pytest.fixture
def window(app,tmp_path,monkeypatch):
    from inference import gui
    monkeypatch.setattr(gui,'OUTPUT_ROOT',tmp_path)
    entries=[ModelEntry('Production',DEFAULT_MODEL,('onnx','pt'),tuple('ABCDEF')),
             ModelEntry('GRU Research',DEFAULT_MODEL,('onnx',),tuple('ABCDEF'),True)]
    w=MainWindow(catalog=entries,session_factory=FakeSession);w.show();app.processEvents()
    yield w
    w.end_session();w.poll_sessions();w.close();app.processEvents()


def test_initialization_and_model_selector_never_open_capture(window,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Hardware opened before START')
    monkeypatch.setattr(cv2,'VideoCapture',forbidden)
    assert window.status.text()=='READY'
    assert len(window.sessions)==0
    assert window.model.count()==2
    assert window.options['backend']=='onnx' and window.options['device']=='cpu'
    assert window.cpu_button.isChecked()


def test_cpu_shortcut_never_switches_to_cuda(window):
    window.cpu_button.click()
    assert window.options['device']=='cpu'
    assert window.cpu_button.isChecked()
    window.options['device']='cuda:0'
    window.cpu_button.setChecked(False)
    window.cpu_button.click()
    assert window.options['device']=='cpu'
    assert window.cpu_button.isChecked()
    assert not window.options['export_video']


def test_start_end_duplicate_start_metrics_and_restart(window,app):
    window.source.setCurrentText('3')
    window.start_session();first=window.sessions[0]
    window.start_session();assert len(window.sessions)==1 and window.sessions[0] is first
    assert first.config['source']=='3'
    assert not window.model.isEnabled()
    window.poll_sessions();app.processEvents();window.poll_sessions()
    assert '20.0' in window.panels[0].metrics.text()
    assert 'A' in window.panels[0].label.text()
    window.end_session();assert window.panels[0].preview.image is None
    window.poll_sessions();assert not window.sessions
    window.model.setCurrentIndex(1);window.start_session()
    assert window.sessions[0] is not first
    assert window.panels[0].preview.display_rate.total==0


def test_single_source_preview_fits_laptop_viewport(window,app):
    window.resize(1100,700)
    window.start_session()
    window.poll_sessions()
    for _ in range(4):
        app.processEvents()
    assert window.intro_cards.isHidden()
    assert window.scroll.verticalScrollBar().maximum()==0
    preview=window.panels[0].preview
    assert preview.height()>=180
    assert window.scroll.viewport().rect().contains(
        preview.mapTo(window.scroll.viewport(),preview.rect().bottomRight()))


def test_multi_source_isolation_roi_and_close(window,app):
    window.extra_sources=['1','2','3'];window.start_session();window.poll_sessions()
    assert len(window.sessions)==4 and len({s.config['run_dir'] for s in window.sessions})==4
    window.resize(960,620);app.processEvents()
    assert window.grid.itemAtPosition(3,0).widget() is window.panels[3]
    window.resize(1400,850);app.processEvents()
    assert window.grid.itemAtPosition(1,1).widget() is window.panels[3]
    window.panels[0].preview.roi=(.1,.2,.8,.9)
    window.panels[0].roi_enabled.setChecked(True)
    assert window.sessions[0].roi==(.1,.2,.8,.9)
    assert all(s.roi is None for s in window.sessions[1:])
    window.panels[0].reset_roi();assert window.sessions[0].roi is None
    window.close();window.poll_sessions();app.processEvents()
    assert not window.sessions and not window.isVisible()


def test_letterbox_roi_mapping_and_visual_drag(app):
    rect=frame_rect(1000,1000,1920,1080)
    assert rect.x()==0 and rect.y()==pytest.approx(218.75)
    assert map_to_frame(QPointF(500,500),rect)==pytest.approx((.5,.5))
    assert map_to_frame(QPointF(-50,-10),rect)==(0,0)
    preview=VideoPreview();preview.resize(800,600)
    preview.set_frame(np.zeros((1080,1920,3),np.uint8),'test',time.monotonic());preview.show();app.processEvents()
    preview.selecting=True;selected=[];preview.roi_selected.connect(selected.append)
    QTest.mousePress(preview,Qt.MouseButton.LeftButton,pos=QPointF(200,200).toPoint())
    QTest.mouseMove(preview,QPointF(600,400).toPoint())
    QTest.mouseRelease(preview,Qt.MouseButton.LeftButton,pos=QPointF(600,400).toPoint())
    assert selected and selected[0]==pytest.approx((.25,5/18,.75,13/18))
    preview.selecting=True
    QTest.mousePress(preview,Qt.MouseButton.LeftButton,pos=QPointF(200,200).toPoint())
    preview.clear('Disconnected during selection')
    QTest.mouseRelease(preview,Qt.MouseButton.LeftButton,pos=QPointF(600,400).toPoint())
    assert len(selected)==1
    preview.close()


def test_no_person_disconnect_and_uncertain_clears_stale_label(app):
    panel=SourcePanel('test',tuple('ABCDEF'),.95)
    frame=np.zeros((120,160,3),np.uint8)
    info=dict(class_id=0,label='A',score=.9,probs=[.9,.02,.02,.02,.02,.02],keypoints_valid=13,track_id=1)
    panel.update_packet(dict(state='RUNNING',info=info,frame=frame,received=time.monotonic()))
    assert 'UNCERTAIN' in panel.label.text()
    panel.update_packet(dict(state='NO PERSON',info={'selection_reason':'ambiguous_candidates'},frame=frame,received=time.monotonic()))
    assert 'Ambiguous' in panel.label.text() and all(b.value()==0 for b in panel.bars)
    panel.update_packet(dict(state='RECONNECTING',info={}))
    assert panel.preview.image is None and panel.label.text()=='No current prediction'
    panel.close()


def test_explicit_backend_failure(window):
    window.model.setCurrentIndex(1);window.options['backend']='pt';window.start_session()
    assert not window.sessions and 'backend unavailable' in window.status.text()


def test_measured_fps_zero_warmup_and_percentiles():
    timing=TimingWindow(3)
    assert timing.report()['yolo_fps'] is None
    for i in range(4):timing.add(dict(yolo_ms=10.,classifier_ms=1.,pipeline_ms=20.),now=1+i*.1)
    report=timing.report()
    assert report['yolo_fps']==100 and report['classifier_fps']==1000
    assert report['pipeline_fps']==pytest.approx(10) and report['window_frames']==3
    assert report['pipeline_ms']['p95']==20
    rate=DisplayRate();assert rate.fps(now=0) is None
    rate.painted(now=1);rate.painted(now=1.1);assert rate.fps(now=1.1)==pytest.approx(10)
    assert rate.fps(now=4) is None


def test_real_spawn_start_end_and_native_hang_containment(app,tmp_path):
    session=SessionProcess({'run_dir':str(tmp_path/'normal')},target=fake_worker)
    session.start()
    try:
        wait_until(app,lambda:session.poll() is not None,seconds=20)
        session.stop();wait_until(app,lambda:(session.poll(),session.finished)[1],seconds=6)
        assert not session.alive() and not session.forced
    finally:
        if session.alive():session.process.terminate();session.process.join(3)
    blocked=SessionProcess({'run_dir':str(tmp_path/'blocked')},target=blocked_worker);blocked.start();blocked.stop()
    try:
        wait_until(app,lambda:(blocked.poll(),blocked.finished)[1],seconds=6)
        assert blocked.forced and not blocked.alive()
    finally:
        if blocked.alive():blocked.process.terminate();blocked.process.join(3)


def test_rapid_start_end_has_no_background_processes(app,tmp_path):
    import multiprocessing as mp
    existing={p.pid for p in mp.active_children()}
    for i in range(3):
        session=SessionProcess({'run_dir':str(tmp_path/str(i))},target=fake_worker)
        session.start();session.stop()
        wait_until(app,lambda:(session.poll(),session.finished)[1],seconds=6)
        assert not session.alive()
    assert {p.pid for p in mp.active_children()}==existing


def test_local_file_eof_reader_and_release(tmp_path):
    path=tmp_path/'synthetic.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30.,(160,120))
    assert writer.isOpened()
    for _ in range(3):writer.write(np.zeros((120,160,3),np.uint8))
    writer.release()
    reader=DesktopReader(str(path),'test')
    assert reader.eof.wait(3)
    assert reader.error is None and reader.seq==3
    reader.close();assert not reader.thread.is_alive()
    temporal=DesktopReader(str(path),'research',True)
    assert temporal.eof.wait(3) and 'timestamps' in temporal.error
    temporal.close()


def test_roi_selector_and_independent_pipeline_states(monkeypatch):
    class Pose:
        def __init__(self,*a,**k):pass
        def track_frame(self,frame):
            return [dict(bbox=[0,0,70,120],track_id=1,keypoints=[[20+j,20+j,.9] for j in range(17)]),
                    dict(bbox=[90,0,160,120],track_id=2,keypoints=[[100+j,20+j,.9] for j in range(17)])]
        def reset_tracking(self):pass
        def close(self):pass
    class Classifier:
        conf_thresh=.35;ema_alpha=.25;labels=list('ABCDEF')
        def predict(self,x):return np.array([.9,.02,.02,.02,.02,.02])
    monkeypatch.setattr(pipeline,'PoseRunner',Pose)
    a=pipeline.PosePipeline('unused',Classifier());b=pipeline.PosePipeline('unused',Classifier())
    frame=np.zeros((120,160,3),np.uint8)
    assert a.process(frame)[0]['selection_reason']=='ambiguous_candidates'
    a.set_roi((0,0,.5,1));info=a.process(frame)[0]
    assert info['class_id']==0 and info['track_id']==1 and info['feature_dim']==52
    assert b.process(frame)[0]['class_id'] is None and b.normalizer.center is None
    a.set_roi((.5,0,1,1));assert a.last_track_id is None and not a.stable.window
    assert a.process(frame)[0]['track_id']==2
    with pytest.raises(ValueError):a.set_roi((1,0,0,1))
    a.close();b.close()


def test_installed_default_bundle_pt_onnx_and_authoritative_labels():
    backends,labels,errors=validate_bundle(DEFAULT_MODEL)
    assert set(backends)=={'pt','onnx'} and not errors
    mapping=json.loads((DEFAULT_MODEL/'label_map.json').read_text(encoding='utf-8'))['labels']
    assert list(labels)==[r['name'] for r in mapping] and 'sang bên' in labels[4]


def test_nonfinite_logits_are_rejected():
    classifier=object.__new__(pipeline.PostureClassifier)
    classifier.is_temporal=False;classifier.input_dim=52;classifier.feature_version='pose13_xy_conf_mask_causal_v2';classifier.backend='onnx'
    class Session:
        def run(self,*args):return [np.full((1,6),np.nan,np.float32)]
    classifier.session=Session()
    with pytest.raises(ValueError,match='non-finite'):classifier.predict(np.zeros(52,np.float32))


def test_two_gru_streams_never_share_warmup_or_track_context(monkeypatch):
    from shared.sequence_data import sequence_contract as make_sequence_contract
    class Pose:
        def __init__(self,*a,**k):self.track_id=1
        def track_frame(self,frame):
            return [dict(bbox=[0,0,160,120],track_id=self.track_id,keypoints=[[20+j,20+j,.9] for j in range(17)])]
        def reset_tracking(self):pass
        def close(self):pass
    class Classifier:
        conf_thresh=.35;ema_alpha=.25;labels=list('ABCDEF');is_temporal=True;sequence_contract=make_sequence_contract(16)
        def predict(self,x):
            assert x.shape==(16,52)
            return np.array([.9,.02,.02,.02,.02,.02])
    monkeypatch.setattr(pipeline,'PoseRunner',Pose)
    a=pipeline.PosePipeline('unused',Classifier());b=pipeline.PosePipeline('unused',Classifier())
    frame=np.zeros((120,160,3),np.uint8)
    rows=[a.process(frame,timestamp_ns=i*50_000_000)[0] for i in range(16)]
    assert all(r['class_id'] is None for r in rows[:15]) and rows[-1]['class_id']==0
    assert len(b.temporal.features)==0
    assert b.process(frame,timestamp_ns=0)[0]['class_id'] is None
    a.pose.track_id=2
    assert a.process(frame,timestamp_ns=800_000_000)[0]['class_id'] is None
    assert len(a.temporal.features)==1 and len(b.temporal.features)==1
    a.reset();assert len(a.temporal.features)==0 and len(b.temporal.features)==1
    a.close();b.close()


def test_disconnect_and_reconnect_reset_runtime_state(tmp_path,monkeypatch):
    import threading
    from types import SimpleNamespace
    from inference import runtime,models
    frame=np.zeros((120,160,3),np.uint8)
    instances=[];packets=[]
    class Reader:
        def __init__(self,*args):self.eof=threading.Event();self.error=None;self.i=0;self.closed=False;instances.append(self)
        def snapshot(self):
            self.i+=1
            if self.i==1:return (1,1,time.monotonic(),frame)
            if self.i==2:return None
            if self.i==3:return (2,2,time.monotonic(),frame)
            self.eof.set();return None
        def close(self):self.closed=True
    class Pipeline:
        def __init__(self,*args,**kwargs):self.resets=0;self.process_epochs=[];self.closed=False;self.pose=SimpleNamespace(track_frame=lambda x:[]);instances.append(self)
        def reset(self):self.resets+=1
        def set_roi(self,value):pass
        def close(self):self.closed=True
        def process(self,frame,**kwargs):
            self.process_epochs.append(self.resets)
            return dict(class_id=0,label='A',score=.9,probs=[.9,.02,.02,.02,.02,.02],pipeline_ms=1.,yolo_ms=.5,classifier_ms=.1),frame
    monkeypatch.setattr(runtime,'DesktopReader',Reader)
    monkeypatch.setattr(runtime,'publish',lambda out,packet:(packets.append(packet),0)[1])
    monkeypatch.setattr(models,'provenance',lambda *args:{})
    monkeypatch.setattr(pipeline,'PostureClassifier',lambda *args:SimpleNamespace(is_temporal=False))
    monkeypatch.setattr(pipeline,'PosePipeline',Pipeline)
    config=dict(run_dir=str(tmp_path),model=str(tmp_path),pose='unused',backend='onnx',device='cpu',threshold=0.,source='0',suppress_native_stderr=False)
    runtime.session_worker(config,threading.Event(),queue.Queue(),queue.Queue())
    p=next(x for x in instances if isinstance(x,Pipeline))
    assert len(p.process_epochs)==2 and p.process_epochs[1]>p.process_epochs[0]
    assert 'RECONNECTING' in [packet['state'] for packet in packets]
    assert all(not packet['info'] for packet in packets if packet['state']=='RECONNECTING')
    assert all(x.closed for x in instances)


def test_fp32_check_uses_actual_delegated_backend_weights():
    from types import SimpleNamespace
    from inference.runtime import runtime_environment
    class Weights:
        def parameters(self):return iter([SimpleNamespace(dtype='torch.float32')])
    active=SimpleNamespace(model=Weights(),device='cuda:0',fp16=False,parameters=lambda:iter([]))
    p=SimpleNamespace(pose=SimpleNamespace(engine=SimpleNamespace(predictor=SimpleNamespace(model=active))))
    c=SimpleNamespace(backend='onnx',session=SimpleNamespace(get_providers=lambda:['CPUExecutionProvider']))
    report=runtime_environment(p,c,'onnx')
    assert report['yolo_dtype']=='torch.float32' and report['classifier_providers']==['CPUExecutionProvider']
    active.fp16=True
    with pytest.raises(ValueError,match='FP32'):runtime_environment(p,c,'onnx')


def test_shared_preview_slot_is_bounded_and_never_waits_on_writer():
    import multiprocessing as mp
    from inference.runtime import FrameSlot
    slot=FrameSlot(mp.get_context('spawn'))
    frame=np.full((120,160,3),42,np.uint8)
    assert slot.write(frame,1)
    np.testing.assert_array_equal(slot.read(1),frame)
    assert slot.read(2) is None
    slot.lock.acquire()
    try:assert slot.read(1) is None
    finally:slot.lock.release()
    large=np.zeros((1440,2560,3),np.uint8)
    assert slot.write(large,2) and slot.read(2).shape==(1080,1920,3)
