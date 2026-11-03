"""Desktop sessions: Qt never owns a blocking camera read or model inference.

One spawned process per stream contains native capture and inference. Native
backend hangs can be terminated without leaving a Qt worker alive. Mutable pose,
ByteTrack, selection and temporal state never cross stream boundaries.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from inference.run import CameraReader, clean_source, video_writer
from inference.telemetry import TimingWindow


class FrameSlot:
    """One shared preview frame; large image bytes never enter a process queue.

    Preview alone is bounded to 1920x1080, preserving aspect ratio. Detection,
    normalization and recording continue to use the original source dimensions.
    The GUI never waits for the writer's lock and checks sequence/metadata match.
    """
    def __init__(self, ctx):
        self.pixels = ctx.RawArray('B',1920*1080*3)
        self.shape = ctx.RawArray('i',3)
        self.sequence = ctx.RawValue('q',0)
        self.lock = ctx.Lock()

    def write(self, frame, sequence):
        h,w=frame.shape[:2]
        scale=min(1.,1920/w,1080/h)
        if scale<1:
            frame=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        if not self.lock.acquire(timeout=.01):
            return False
        try:
            flat=np.frombuffer(self.pixels,dtype=np.uint8)
            flat[:frame.size]=frame.reshape(-1)
            self.shape[:]=frame.shape
            self.sequence.value=sequence
        finally:
            self.lock.release()
        return True

    def read(self, sequence):
        if not self.lock.acquire(False):
            return None
        try:
            if self.sequence.value!=sequence:
                return None
            shape=tuple(self.shape[:])
            if min(shape)<=0:
                return None
            return np.frombuffer(self.pixels,dtype=np.uint8,count=int(np.prod(shape))).reshape(shape).copy()
        finally:
            self.lock.release()


class DesktopReader(CameraReader):
    """Reuse network/webcam acquisition; local development files end, never loop."""
    def __init__(self, source, name, temporal=False):
        self.eof = threading.Event()
        self.error = None
        self.temporal = temporal
        super().__init__(source, name, open_timeout_ms=2500, read_timeout_ms=1500)

    def _work(self):
        if not isinstance(self.source, str) or not Path(self.source).is_file():
            super()._work()
            return
        cap = None
        try:
            ticks = None
            sidecar = Path(self.source).with_name(Path(self.source).stem.replace('_raw', '')+'_timestamps.csv')
            if sidecar.is_file():
                with sidecar.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                if [int(r['writer_frame_index']) for r in rows] != list(range(len(rows))):
                    raise ValueError('Capture timestamp sidecar order mismatch')
                ticks = [int(r['host_recv_monotonic_ns']) for r in rows]
            elif self.temporal:
                raise ValueError('Research GRU video requires original *_timestamps.csv beside the raw video')
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise OSError('Cannot decode local video; choose a supported MP4')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 120 else 25.
            self.file_fps = fps
            self.generation = 1
            due = time.monotonic()
            while not self.stop.is_set():
                if self.stop.wait(max(0., due-time.monotonic())):
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if ticks is not None and self.seq >= len(ticks):
                    raise ValueError('Video has more frames than timestamp sidecar')
                received = time.monotonic()
                capture_ns = ticks[self.seq] if ticks is not None else int(received*1e9)
                with self.lock:
                    self.seq += 1
                    self.latest = (1, self.seq, received, frame, capture_ns)
                due = max(due+1/fps, received)
            if not self.stop.is_set() and ticks is not None and self.seq != len(ticks):
                raise ValueError('Video and timestamp sidecar frame counts differ')
        except Exception as error:
            self.error = str(error)  # file errors only, never network credentials
        finally:
            if cap is not None:
                cap.release()
            self.eof.set()


def publish(outbox, packet):
    """Bounded mailbox: replace a queued display frame, never build latency."""
    skipped = 0
    try:
        outbox.get_nowait()
        skipped = 1
    except queue.Empty:
        pass
    try:
        outbox.put_nowait(packet)
        return skipped
    except queue.Full:
        return skipped+1


def runtime_environment(pipeline, classifier, backend):
    """Ultralytics AutoBackend delegates weights to .model, not its own parameters."""
    active = getattr(getattr(getattr(pipeline.pose,'engine',None),'predictor',None),'model',None)
    weights = getattr(active,'model',active)
    parameter = next(weights.parameters(),None) if hasattr(weights,'parameters') else None
    environment = dict(yolo_device=str(getattr(active,'device','unknown')),
                       yolo_fp16=bool(getattr(active,'fp16',False)),
                       yolo_dtype=str(parameter.dtype) if parameter is not None else None,
                       classifier_backend=getattr(classifier,'backend',backend),
                       classifier_device='cpu',
                       classifier_providers=classifier.session.get_providers() if hasattr(classifier,'session') else None)
    if environment['yolo_fp16'] or (active is not None and environment['yolo_dtype']!='torch.float32'):
        raise ValueError('YOLO must run verified FP32; check Ultralytics backend settings')
    return environment


def session_worker(config, stop, outbox, controls, frame_slot=None):
    # Native FFmpeg diagnostics can include URL credentials. Keep native stderr
    # private; all actionable errors go to the sanitized status mailbox.
    if config.get('suppress_native_stderr', True):
        with open(os.devnull, 'w') as sink:
            os.dup2(sink.fileno(), 2)
    from inference.models import provenance
    from inference.pipeline import PosePipeline, PostureClassifier
    import torch
    torch.set_num_threads(1)

    folder = Path(config['run_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    shutdown_trace = []
    def mark_shutdown(stage, **details):
        shutdown_trace.append(dict(stage=stage,monotonic=time.monotonic(),**details))
        (folder/'shutdown_trace.json').write_text(json.dumps(shutdown_trace,indent=2),encoding='utf-8')
    reader = pipeline = writer = csv_file = json_file = None
    stats = TimingWindow()
    started = time.monotonic()
    dropped = display_skipped = 0
    last_key = (-1, 0)
    last_received = None
    stale_reset = False
    state = 'STARTING'
    error_text = None
    packet_no = 0
    roi_epoch = 0
    stage = 'model_loading'
    try:
        publish(outbox, dict(state=state, info={}))
        classifier = PostureClassifier(Path(config['model']), config['backend'], 'cpu')
        stage = 'pose_initialization'
        pipeline = PosePipeline(config['pose'], classifier, config['device'],
                                min_conf=config['threshold'])
        pipeline.set_roi(config.get('roi'))
        # Initialize CUDA/Ultralytics before acquiring the source so short files
        # and live streams do not age in the mailbox during model warmup.
        pipeline.pose.track_frame(np.zeros((480,640,3),np.uint8))
        pipeline.reset()
        environment = runtime_environment(pipeline,classifier,config['backend'])
        (folder/'runtime_environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
        safe_config = {k: v for k, v in config.items() if k != 'source'}
        safe_config['source'] = clean_source(str(config['source']))
        safe_config['classifier_device'] = 'cpu'
        safe_config['precision'] = 'FP32'
        (folder/'run_config.json').write_text(json.dumps(safe_config, indent=2), encoding='utf-8')
        (folder/'provenance.json').write_text(json.dumps(provenance(config['model'], config['pose']), indent=2), encoding='utf-8')
        if config.get('export_predictions'):
            csv_file = (folder/'predictions.csv').open('w', encoding='utf-8-sig', newline='')
            columns = ['frame','capture_ns','reference_label','class_id','label','score','track_id',
                       'keypoints_valid','pose_quality','selection_reason','temporal_state','feature_dim',
                       'yolo_ms','classifier_ms','pipeline_ms','freshness_ms'] + [f'p{i}' for i in range(6)]
            csv_log = csv.DictWriter(csv_file, fieldnames=columns)
            csv_log.writeheader()
            json_file = (folder/'predictions.jsonl').open('w', encoding='utf-8')
        reader = DesktopReader(config['source'], 'source', classifier.is_temporal)
        stage = 'capture_or_processing'
        while not stop.is_set():
            roi_changed = False
            try:
                while True:
                    command = controls.get_nowait()
                    if command['type'] == 'roi':
                        pipeline.set_roi(command['value'])
                        roi_epoch = command.get('epoch', roi_epoch+1)
                        roi_changed = True
            except queue.Empty:
                pass
            if roi_changed:
                publish(outbox, dict(state='STARTING', info={}))
            item = reader.snapshot()
            now = time.monotonic()
            if item is None or (item[0], item[1]) == last_key:
                if reader.eof.is_set():
                    if reader.error:
                        raise ValueError(reader.error)
                    state = 'STOPPED'
                    break
                if item is None or (last_received is not None and now-last_received > 2):
                    if not stale_reset:
                        pipeline.reset()
                        publish(outbox, dict(state='RECONNECTING', info={}))
                        stale_reset = True
                    if now - (last_received or started) > config.get('connect_deadline', 30):
                        raise OSError('Source unavailable. Check camera index, DroidCam connection, URL and OpenCV codec support; then START again.')
                stop.wait(.005)
                continue
            generation, seq, received, frame = item[:4]
            if now-received > 2:
                if not stale_reset:
                    pipeline.reset()
                    publish(outbox, dict(state='RECONNECTING', info={}))
                    stale_reset = True
                stop.wait(.005)
                continue
            if generation != last_key[0]:
                pipeline.reset()
            dropped += max(0, seq-last_key[1]-1)
            last_key = (generation, seq)
            stale_reset = False
            last_received = received
            capture_ns = item[4] if len(item)>4 else int(received*1e9)
            info, annotated = pipeline.process(frame, timestamp_ns=capture_ns)
            info['freshness_ms'] = (time.monotonic()-received)*1000
            stats.add(info)
            state = 'NO PERSON' if info['class_id'] is None and not info.get('temporal_state') else 'RUNNING'
            if config.get('export_video'):
                if writer is None:
                    writer = video_writer(folder/'annotated.mp4', frame.shape[1], frame.shape[0], getattr(reader, 'file_fps', 25.))
                # Qt uses the authoritative Unicode label; OpenCV output uses
                # class ID and score so exported text is legible in any locale.
                label = 'NO POSE' if info['class_id'] is None else f"Class {info['class_id']}  {info['score']:.1%}"
                recorded = annotated.copy()
                cv2.putText(recorded, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                writer.write(recorded)
            if json_file is not None:
                row = dict(frame=seq, capture_ns=capture_ns, reference_label=config.get('reference_label'), **info)
                json_file.write(json.dumps(row, ensure_ascii=False)+'\n')
                flat = {k: row.get(k) for k in columns}
                flat.update({f'p{i}': p for i,p in enumerate(info['probs'] or [])})
                csv_log.writerow(flat)
            packet_no += 1
            image_packet = {'frame':annotated}
            if frame_slot is not None:
                if not frame_slot.write(annotated,packet_no):
                    display_skipped+=1
                    continue
                image_packet={'frame_seq':packet_no}
            display_skipped += publish(outbox, dict(state=state, info=info, **image_packet,
                    roi_epoch=roi_epoch,
                    received=received, packet_no=packet_no, telemetry=stats.report(),
                    capture_skipped=dropped, display_mailbox_skipped=display_skipped))
        state = 'STOPPED'
    except Exception as error:
        state = 'ERROR'
        # No raw backend exception containing a source URL can reach logs/UI.
        error_text = f'{type(error).__name__} during {stage}: {error}'
        source = str(config.get('source', ''))
        if source.lower().startswith(('rtsp:', 'rtsps:', 'http:', 'https:')):
            if source in error_text or '://' in error_text:
                error_text = f'{type(error).__name__}: stream failed; verify connection and selected backend. URL details suppressed to protect credentials.'
    finally:
        mark_shutdown('cleanup_started')
        if reader is not None:
            reader.close()
        mark_shutdown('reader_closed',reader_alive=bool(reader and hasattr(reader,'thread') and reader.thread.is_alive()))
        if writer is not None:
            writer.release()
        for handle in (csv_file, json_file):
            if handle is not None:
                handle.close()
        if pipeline is not None:
            pipeline.close()
        mark_shutdown('pipeline_closed')
        elapsed = time.monotonic()-started
        report = dict(state=state, error=error_text, wall_seconds=elapsed,
                      session_wall_fps=stats.total/elapsed if elapsed>0 else None,
                      capture_skipped=dropped, display_mailbox_skipped=display_skipped,
                      sampling='Last 120 processed frames after synthetic model initialization; Live FPS belongs to Qt display.json',
                      **stats.report())
        (folder/'performance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        publish(outbox, dict(state=state, error=error_text, info={}, telemetry=report))
        mark_shutdown('terminal_published',threads=[dict(name=t.name,daemon=t.daemon) for t in threading.enumerate()])
    if frame_slot is not None:
        # This is an owned, isolated native-runtime process. All capture, video,
        # log and pipeline resources above have already closed. Flush the small
        # terminal message, then exit explicitly: Windows CUDA/ORT DLL teardown
        # after Python finalization can otherwise stall despite no live reader.
        # The OS releases the CUDA context and shared-memory handles; the parent
        # joins the process. Direct/in-process test callers take the normal path.
        outbox.close()
        outbox.join_thread()
        mark_shutdown('owned_process_exit_after_io_cleanup')
        os._exit(0 if state!='ERROR' else 1)


class SessionProcess:
    """Non-blocking controller polled by Qt; hard deadline contains native hangs."""
    def __init__(self, config, target=session_worker):
        ctx = mp.get_context('spawn')
        self.config = config
        self.stop_event = ctx.Event()
        self.outbox = ctx.Queue(maxsize=1)
        self.controls = ctx.Queue(maxsize=16)
        self.frame_slot=FrameSlot(ctx) if target is session_worker else None
        args=(config,self.stop_event,self.outbox,self.controls)
        if self.frame_slot is not None:args+= (self.frame_slot,)
        self.process = ctx.Process(target=target, args=args, daemon=True)
        self.deadline = None
        self.forced = False
        self.last_packet = None
        self.finished = False
        self.roi_epoch = 0

    def start(self):
        try:
            self.process.start()
        except Exception:
            self.finished = True
            for mailbox in (self.outbox,self.controls):
                mailbox.cancel_join_thread()
                mailbox.close()
            raise

    def stop(self):
        self.stop_event.set()
        if self.deadline is None:
            self.deadline = time.monotonic()+3.

    def set_roi(self, value):
        try:
            epoch = self.roi_epoch+1
            self.controls.put_nowait(dict(type='roi', value=value, epoch=epoch))
            self.roi_epoch = epoch
        except queue.Full:
            raise RuntimeError('ROI update queue busy; wait for the current update')

    def poll(self):
        if self.finished:
            return None
        packet = None
        try:
            while True:
                packet = self.outbox.get_nowait()
        except (queue.Empty, ValueError, OSError):
            pass
        if packet is not None and 'frame' in packet and packet.get('roi_epoch',0)<self.roi_epoch:
            packet = None
        if packet is not None and 'frame_seq' in packet:
            frame=self.frame_slot.read(packet['frame_seq'])
            if frame is None or packet.get('roi_epoch',0)<self.roi_epoch:
                packet=None
            else:
                packet['frame']=frame
        if packet is not None:
            self.last_packet = packet
        if self.deadline is not None and self.process.is_alive() and time.monotonic() >= self.deadline:
            self.process.terminate()
            self.forced = True
            self.deadline = time.monotonic()+1.
        if not self.process.is_alive() and self.process.pid is not None and not self.finished:
            self.process.join(timeout=0)
            self.finished = True
            state = self.last_packet.get('state') if self.last_packet else None
            if self.forced or state not in ('STOPPED', 'ERROR'):
                packet = dict(state='STOPPED' if self.stop_event.is_set() else 'ERROR', info={},
                              error='Native worker stopped at shutdown deadline; exports may be incomplete.' if self.forced else 'Worker exited before completing; check CUDA memory and model dependencies.')
            folder = Path(self.config['run_dir'])
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'shutdown.json').write_text(json.dumps(dict(forced=self.forced,exitcode=self.process.exitcode)),encoding='utf-8')
            self.process.close()
            for mailbox in (self.outbox, self.controls):
                mailbox.cancel_join_thread()
                mailbox.close()
        return packet

    def alive(self):
        return not self.finished and self.process.is_alive()
