"""PoseForge Windows desktop inference. Launch: python -m inference.gui."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget)

from inference.theme import DESKTOP_STYLE
from inference.models import OUTPUT_ROOT, POSE_MODEL, discover_models
from inference.run import clean_source, make_run_dir
from inference.runtime import SessionProcess
from inference.telemetry import DisplayRate


def frame_rect(width, height, frame_width, frame_height):
    if min(width,height,frame_width,frame_height) <= 0:
        return QRectF()
    scale = min(width/frame_width, height/frame_height)
    w, h = frame_width*scale, frame_height*scale
    return QRectF(max(0.,(width-w)/2), max(0.,(height-h)/2), w, h)


def map_to_frame(point, rect):
    """Letterbox-aware normalized coordinates; dragging outside clamps to edge."""
    if rect.isEmpty():
        return None
    return (max(0.,min(1.,(point.x()-rect.x())/rect.width())),
            max(0.,min(1.,(point.y()-rect.y())/rect.height())))


class VideoPreview(QWidget):
    roi_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240,180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.image = None
        self.text = 'Select a source, then START'
        self.roi = None
        self.roi_visible = True
        self.selecting = False
        self.anchor = self.drag = None
        self.received = None
        self.pending_paint = False
        self.display_rate = DisplayRate()
        self.freshness = deque(maxlen=120)

    def content_rect(self):
        return frame_rect(self.width(),self.height(),self.image.width(),self.image.height()) if self.image is not None else QRectF()

    def set_frame(self, frame, text, received):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.text, self.received = text, received
        self.pending_paint = True
        self.update()

    def clear(self, text='No current prediction'):
        self.image = None
        self.anchor = self.drag = None
        self.pending_paint = False
        self.text = text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#e8eef7'))
        rect = self.content_rect()
        if self.image is not None:
            painter.drawImage(rect,self.image)
            if self.pending_paint:
                self.display_rate.painted()
                if self.received is not None:
                    self.freshness.append((time.monotonic()-self.received)*1000)
                self.pending_paint = False
            roi = (*self.anchor,*self.drag) if self.anchor is not None and self.drag is not None else self.roi if self.roi_visible else None
            if roi is not None:
                x1,y1,x2,y2 = roi
                painter.setPen(QPen(QColor('#3975e8'),2))
                painter.drawRect(QRectF(rect.x()+min(x1,x2)*rect.width(),rect.y()+min(y1,y2)*rect.height(),abs(x2-x1)*rect.width(),abs(y2-y1)*rect.height()))
        painter.fillRect(QRectF(0,0,self.width(),42),QColor(255,255,255,235))
        painter.setPen(QColor('#24334b'))
        painter.setFont(QFont('Segoe UI',11))
        painter.drawText(QRectF(12,0,self.width()-24,42),Qt.AlignmentFlag.AlignVCenter,self.text)
        painter.end()

    def mousePressEvent(self, event):
        if self.selecting and self.content_rect().contains(event.position()):
            self.anchor = self.drag = map_to_frame(event.position(),self.content_rect())
            self.update()

    def mouseMoveEvent(self, event):
        if self.anchor is not None:
            self.drag = map_to_frame(event.position(),self.content_rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.anchor is not None:
            end = map_to_frame(event.position(),self.content_rect())
            x1,y1 = self.anchor; x2,y2 = end
            value = (min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))
            self.anchor = self.drag = None
            if value[2]-value[0] > .01 and value[3]-value[1] > .01:
                self.roi = value
                self.selecting = False
                self.unsetCursor()
                self.roi_selected.emit(value)
            self.update()


class SourcePanel(QWidget):
    roi_changed = Signal(object)

    def __init__(self, title, labels, threshold, compact=False):
        super().__init__()
        self.setObjectName('sourcePanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.labels = labels
        self.threshold = threshold
        self.preview = VideoPreview()
        if compact:self.preview.setMinimumHeight(120)
        self.label = QLabel('Chưa có kết quả');self.label.setObjectName('prediction')
        self.label.setWordWrap(True)
        self.pose = QLabel('Pose: missing');self.pose.setObjectName('poseInfo');self.pose.setWordWrap(True)
        self.metrics = QLabel('YOLO Model FPS: —   |   Classifier FPS: —\nPipeline FPS: —   |   Live FPS: —')
        self.metrics.setWordWrap(True);self.metrics.setObjectName('metrics')
        self.details = QLabel('Latency: waiting for measured frames')
        self.details.setWordWrap(True)
        self.bars = []
        prediction = QWidget();prediction.setObjectName('predictionPanel'); right = QVBoxLayout(prediction)
        right.setContentsMargins(12,12,12,12);right.setSpacing(10)
        prediction.setMaximumWidth(180 if compact else 210)
        right.addWidget(self.label);right.addWidget(self.pose)
        show_probs=QCheckBox('Xác suất từng tư thế');show_probs.setChecked(not compact);right.addWidget(show_probs)
        probabilities=QWidget();prob_layout=QVBoxLayout(probabilities);prob_layout.setContentsMargins(0,0,0,0)
        for text in labels:
            name = QLabel(text);name.setWordWrap(True);name.setObjectName('probabilityName')
            bar = QProgressBar();bar.setRange(0,1000);bar.setValue(0);bar.setFormat('—')
            prob_layout.addWidget(name);prob_layout.addWidget(bar);self.bars.append(bar)
        right.addWidget(probabilities);probabilities.setVisible(not compact);show_probs.toggled.connect(probabilities.setVisible)
        right.addStretch()
        # Long labels/probabilities scroll independently instead of forcing the
        # camera panel taller than the available viewport.
        prediction_scroll = QScrollArea()
        prediction_scroll.setWidgetResizable(True)
        prediction_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        prediction_scroll.setMinimumHeight(0)
        prediction_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        prediction_scroll.setMinimumWidth(160 if compact else 180)
        prediction_scroll.setMaximumWidth(180 if compact else 210)
        prediction_scroll.setWidget(prediction)
        layout = QVBoxLayout(self);layout.setContentsMargins(12,12,12,10);layout.setSpacing(10)
        title_label = QLabel(title);title_label.setObjectName('sourceTitle');title_label.setWordWrap(True);layout.addWidget(title_label)
        body = QHBoxLayout();body.addWidget(self.preview,4);body.addWidget(prediction_scroll,1);layout.addLayout(body,1)
        roi_row = QHBoxLayout()
        self.roi_enabled = QCheckBox('Bật ROI')
        select = QPushButton('Chọn vùng ROI');reset = QPushButton('Đặt lại ROI')
        select.clicked.connect(self.select_roi);reset.clicked.connect(self.reset_roi)
        self.roi_enabled.toggled.connect(self.toggle_roi)
        roi_row.addWidget(self.roi_enabled);roi_row.addWidget(select);roi_row.addWidget(reset);roi_row.addStretch()
        show_details=QCheckBox('Chi tiết hiệu suất');show_details.setChecked(False)
        show_details.toggled.connect(self.details.setVisible);self.details.setVisible(False);roi_row.addWidget(show_details)
        self.expand_preview = QPushButton('⛶  Mở rộng hình')
        self.expand_preview.setCheckable(True)
        self.expand_preview.setToolTip('Ẩn bảng kết quả để tăng diện tích camera')
        def expand_preview(expanded):
            prediction_scroll.setVisible(not expanded)
            self.expand_preview.setText('⛶  Hiện kết quả' if expanded else '⛶  Mở rộng hình')
        self.expand_preview.toggled.connect(expand_preview)
        roi_row.addWidget(self.expand_preview)
        layout.addLayout(roi_row);layout.addWidget(self.metrics);layout.addWidget(self.details)
        self.preview.roi_selected.connect(self.selected_roi)
        self.telemetry = {};self.capture_skipped = 0

    def select_roi(self):
        self.preview.selecting = True
        self.preview.setCursor(Qt.CursorShape.CrossCursor)

    def selected_roi(self, value):
        was_enabled = self.roi_enabled.isChecked()
        self.roi_enabled.setChecked(True)
        if was_enabled:
            self.roi_changed.emit(value)
        self.clear_prediction('ROI updated; waiting for a new pose')

    def toggle_roi(self, checked):
        self.preview.roi_visible = checked
        self.roi_changed.emit(self.preview.roi if checked else None)
        self.clear_prediction('ROI changed; waiting for a new pose')
        self.preview.update()

    def reset_roi(self):
        self.preview.roi = None
        self.preview.selecting = False
        self.preview.unsetCursor()
        if self.roi_enabled.isChecked():
            self.roi_enabled.setChecked(False)
        else:
            self.roi_changed.emit(None)
        self.preview.update()

    def clear_prediction(self, message='No current prediction'):
        self.label.setText(message);self.pose.setText('Pose: missing')
        for bar in self.bars:
            bar.setValue(0);bar.setFormat('—')

    def update_packet(self, packet):
        info = packet.get('info',{})
        if packet.get('telemetry'):
            self.telemetry = packet['telemetry']
        self.capture_skipped = packet.get('capture_skipped',self.capture_skipped)
        if 'frame' not in packet:
            self.preview.clear(packet.get('error') or ('GRU warmup' if info.get('temporal_state') else 'No current prediction'))
            self.clear_prediction()
            return
        if info.get('probs') is None:
            text = ('Không đủ dữ liệu tư thế — đưa vai và tay vào khung hình'
                    if info.get('pose_quality') == 'insufficient_upper_body' else
                    'GRU warmup / invalid context' if info.get('temporal_state') else
                    'Ambiguous people — no prediction' if info.get('selection_reason') == 'ambiguous_candidates' else 'No valid pose')
            self.clear_prediction(text)
        else:
            text = (f"UNCERTAIN • {info['score']:.1%}" if self.threshold and info['score'] < self.threshold else
                    f"{info['label']} • {info['score']:.1%}")
            self.label.setText(text)
            self.pose.setText(f"Pose: {info['keypoints_valid']}/13 joints • Track {info.get('track_id')}\n{info.get('selection_reason','')}")
            for bar, probability in zip(self.bars,info['probs']):
                bar.setValue(round(probability*1000));bar.setFormat(f'{probability:.1%}')
        self.preview.set_frame(packet['frame'],text,packet['received'])

    def refresh_metrics(self):
        def fmt(value):return '—' if value is None else f'{value:.1f}'
        t = self.telemetry
        self.metrics.setText(f"YOLO Model FPS: {fmt(t.get('yolo_fps'))}   |   Classifier FPS: {fmt(t.get('classifier_fps'))}\nPipeline FPS: {fmt(t.get('pipeline_fps'))}   |   Live FPS: {fmt(self.preview.display_rate.fps())}")
        latency = []
        for name,key in [('YOLO','yolo_ms'),('Classifier','classifier_ms'),('Pipeline','pipeline_ms')]:
            value = t.get(key)
            latency.append(f"{name} {fmt(value['median'] if value else None)}/{fmt(value['p95'] if value else None)}")
        freshness = self.preview.freshness[-1] if self.preview.freshness else None
        skipped = max(0,t.get('processed_frames',0)-self.preview.display_rate.total)
        self.details.setText('Median/p95 ms: '+ ' • '.join(latency)+f'\nDisplay freshness: {fmt(freshness)} ms • Capture skipped: {self.capture_skipped} • Processed but not painted: {skipped}')


class AdvancedSettings(QDialog):
    def __init__(self, options, labels, parent):
        super().__init__(parent);self.setWindowTitle('Cài đặt nâng cao');self.setMinimumWidth(520)
        layout = QFormLayout(self)
        self.backend = QComboBox();self.backend.addItems(['onnx','pt']);self.backend.setCurrentText(options['backend'])
        self.device = QComboBox();self.device.addItems(['cuda:0','cpu']);self.device.setCurrentText(options['device'])
        self.threshold = QDoubleSpinBox();self.threshold.setRange(0,1);self.threshold.setSingleStep(.05);self.threshold.setValue(options['threshold'])
        self.video = QCheckBox('Lưu video MP4 có chú thích');self.video.setChecked(options['export_video'])
        self.logs = QCheckBox('Lưu kết quả CSV + JSONL');self.logs.setChecked(options['export_predictions'])
        self.reference = QComboBox();self.reference.addItem('Unlabeled',None)
        for i,label in enumerate(labels):self.reference.addItem(label,i)
        self.reference.setCurrentIndex(0 if options.get('reference_label') is None else options['reference_label']+1)
        layout.addRow('Bộ phân loại (CPU, FP32)',self.backend)
        layout.addRow('Thiết bị YOLO (FP32)',self.device)
        layout.addRow('Ngưỡng chưa chắc chắn (0: tắt)',self.threshold)
        layout.addRow(self.video);layout.addRow(self.logs)
        layout.addRow('Nhãn tham chiếu (tùy chọn)',self.reference)
        note=QLabel('Reference labels are operator annotations, not verified accuracy.\nEND the session before changing models/settings. No keypoint smoothing is added.');note.setWordWrap(True);layout.addRow(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addRow(buttons)

    def values(self):
        return dict(backend=self.backend.currentText(),device=self.device.currentText(),threshold=self.threshold.value(),
                    export_video=self.video.isChecked(),export_predictions=self.logs.isChecked(),reference_label=self.reference.currentData())


class MainWindow(QMainWindow):
    def __init__(self, catalog=None, session_factory=SessionProcess):
        super().__init__();self.setWindowTitle('PoseForge • Real-time posture inference');self.resize(1480,900)
        self.session_factory=session_factory
        self.catalog,self.catalog_errors=discover_models() if catalog is None else (catalog,[])
        self.sessions=[];self.panels=[];self.states=[];self.closing=False;self.stopping=False
        self.options=dict(backend='onnx',device='cpu',threshold=0.,export_video=False,export_predictions=False,reference_label=None)
        self.run_dir=None
        central=QWidget();self.setCentralWidget(central);layout=QVBoxLayout(central)
        controls=QGridLayout()
        self.source=QComboBox();self.source.setEditable(True);self.source.addItems(['0','1','2','3']);self.source.setMinimumWidth(170)
        self.source.setToolTip('Camera index, DroidCam/RTSP/HTTP URL, or local MP4 path; hardware opens only on START')
        self.browse=QPushButton('Video…');self.browse.clicked.connect(self.browse_video)
        self.model=QComboBox()
        for entry in self.catalog:self.model.addItem(entry.name,entry)
        self.model.setMinimumWidth(260)
        self.start_button=QPushButton('Bắt đầu');self.start_button.clicked.connect(self.start_session)
        self.end_button=QPushButton('Kết thúc');self.end_button.clicked.connect(self.end_session);self.end_button.setEnabled(False)
        self.add_button=QPushButton('+ Thêm nguồn');self.add_button.clicked.connect(self.add_source)
        self.remove_button=QPushButton('Bỏ nguồn cuối');self.remove_button.clicked.connect(self.remove_source)
        self.advanced=QPushButton('Cài đặt nâng cao');self.advanced.clicked.connect(self.advanced_settings)
        from inference.workspace import build_workspace
        build_workspace(self, layout)
        self.timer=QTimer(self);self.timer.setInterval(16);self.timer.timeout.connect(self.poll_sessions);self.timer.start()
        self.model.currentIndexChanged.connect(self.model_changed)
        self.setStyleSheet(DESKTOP_STYLE)
        if not self.catalog:
            self.start_button.setEnabled(False);self.status.setText('ERROR • No validated model bundles. '+ '; '.join(self.catalog_errors))
        elif self.catalog_errors:
            self.status.setText('READY • Some model backends unavailable; '+ '; '.join(self.catalog_errors))

    def select_cpu(self, checked):
        self.options['device']='cpu'
        self.cpu_button.setChecked(True)

    def model_changed(self):
        # Settings are locked during a session. Each subsequent START constructs
        # fresh pipeline state, including any research temporal buffers.
        for panel in self.panels:panel.clear_prediction()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'panels'):
            self.relayout_panels()

    def relayout_panels(self):
        columns=2 if self.width()>=1250 and len(self.panels)>1 else 1
        for i,panel in enumerate(self.panels):
            self.grid.addWidget(panel,i//columns,i%columns)

    def browse_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Select local development video','','Videos (*.mp4 *.avi *.mkv *.mov);;All files (*)')
        if path:self.source.setCurrentText(path)

    def add_source(self):
        if len(self.extra_sources)>=3:
            self.status.setText('READY • Maximum four sources');return
        value,ok=QInputDialog.getText(self,'Add Source','Camera index, local MP4 path, RTSP or HTTP URL:')
        if ok and value.strip():
            self.extra_sources.append(value.strip())
            self.source_names.setText('Additional sources: '+ ' | '.join(f'{i+2}: {clean_source(s)}' for i,s in enumerate(self.extra_sources)))

    def remove_source(self):
        if not self.sessions and self.extra_sources:
            self.extra_sources.pop();self.source_names.setText('Additional sources: '+ ' | '.join(clean_source(s) for s in self.extra_sources))

    def advanced_settings(self):
        entry=self.model.currentData()
        if entry is None:return
        dialog=AdvancedSettings(self.options,entry.labels,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            self.options=dialog.values()
            self.cpu_button.setChecked(self.options['device']=='cpu')

    def lock_controls(self, running):
        for widget in (self.source,self.browse,self.model,self.add_button,self.remove_button,self.advanced,self.cpu_button):widget.setEnabled(not running)
        self.start_button.setEnabled(not running and bool(self.catalog));self.end_button.setEnabled(running)

    def start_session(self):
        if self.sessions:return
        entry=self.model.currentData()
        if entry is None:return
        if self.options['backend'] not in entry.backends:
            self.status.setText('ERROR • Selected backend unavailable. Choose a validated backend in Advanced Settings.');return
        source=self.source.currentText().strip()
        if not source:self.status.setText('ERROR • Enter a camera index, stream URL or local file.');return
        sources=[source]+self.extra_sources
        for value in sources:
            if not value.isdigit() and not value.lower().startswith(('rtsp://','rtsps://','http://','https://')) and not Path(value).is_file():
                self.status.setText('ERROR • Source file does not exist. Choose an MP4, camera index, or supported stream URL.');return
        self.run_dir=make_run_dir(OUTPUT_ROOT,datetime.now().strftime('desktop_%Y%m%d_%H%M%S'))
        while self.grid.count():
            item=self.grid.takeAt(0)
            if item.widget():item.widget().hide();item.widget().deleteLater()
        self.panels=[];self.states=['STARTING']*len(sources);self.sessions=[];self.stopping=False
        self.intro_cards.hide()
        self.sidebar_toggle.setChecked(False)
        self.scroll.verticalScrollBar().setValue(0)
        self.lock_controls(True);self.status.setText('STARTING')
        self.output.setText('Session: '+str(self.run_dir)+' • Video '+('ON' if self.options['export_video'] else 'OFF'))
        for i,source in enumerate(sources):
            panel=SourcePanel(f'Source {i+1}: {clean_source(source)} • {entry.name} • {self.options["backend"].upper()} CPU',entry.labels,self.options['threshold'],compact=len(sources)>1)
            panel.roi_changed.connect(lambda value,index=i:self.update_roi(index,value))
            self.grid.addWidget(panel, i//2 if len(sources)>2 else 0, i%2 if len(sources)>1 else 0)
            self.panels.append(panel)
            config=dict(source=source,model=str(entry.path),pose=str(POSE_MODEL),run_dir=str(self.run_dir/f'source_{i+1:02d}'),**self.options)
            session=self.session_factory(config);self.sessions.append(session)
            try:session.start()
            except Exception as error:
                self.states[i]='ERROR';self.status.setText('ERROR • Unable to start worker: '+str(error));self.end_session();return
        self.relayout_panels()

    def update_roi(self,index,value):
        if index<len(self.sessions) and self.sessions[index].alive():
            try:
                self.sessions[index].set_roi(value)
                self.panels[index].preview.clear('ROI updated; waiting for current pose')
            except RuntimeError as error:
                self.status.setText('ERROR • '+str(error))

    def end_session(self):
        if not self.sessions:return
        self.stopping=True;self.end_button.setEnabled(False)
        self.status.setText('STOPPED • Releasing sources…')
        for session in self.sessions:session.stop()
        for panel in self.panels:panel.preview.clear('Session ended');panel.clear_prediction()

    def poll_sessions(self):
        if not self.sessions:return
        for i,session in enumerate(self.sessions):
            packet=session.poll()
            if packet is not None:
                self.states[i]=packet['state']
                if not self.stopping:self.panels[i].update_packet(packet)
                if packet.get('error'):self.states[i]='ERROR • '+packet['error']
            self.panels[i].refresh_metrics()
        if not self.stopping:
            state=('ERROR' if any(s.startswith('ERROR') for s in self.states) else
                   'RUNNING' if 'RUNNING' in self.states else
                   'RECONNECTING' if 'RECONNECTING' in self.states else
                   'STARTING' if 'STARTING' in self.states else
                   'NO PERSON' if 'NO PERSON' in self.states else 'STOPPED')
            detail=' | '.join(f'Source {i+1}: {s}' for i,s in enumerate(self.states))
            if len(self.panels)>1 and any(p.telemetry.get('processed_frames',0)>30 and
                    (p.telemetry.get('pipeline_fps') or 0)<15 for p in self.panels):
                detail+=' • Throughput limited; reduce source count for a faster preview'
            self.status.setText(state+' • '+detail)
        if all(s.finished for s in self.sessions):
            try:
                self.save_display_report()
            except OSError as error:
                self.status.setText('ERROR • Session stopped but display report could not be saved: '+str(error))
            self.sessions=[];self.lock_controls(False)
            if self.stopping:self.status.setText('STOPPED • Sources released')
            if self.closing:self.close()

    def save_display_report(self):
        import numpy as np
        for i,panel in enumerate(self.panels):
            values=list(panel.preview.freshness)
            report=dict(painted_frames=panel.preview.display_rate.total,live_fps_at_end=panel.preview.display_rate.fps(),
                        average_live_fps=panel.preview.display_rate.average_fps(),
                        freshness_median_ms=float(np.median(values)) if values else None,
                        freshness_p95_ms=float(np.percentile(values,95)) if values else None,
                        measurement='Unique frames painted by Qt; compositor/monitor scanout and camera exposure are not measured',
                        forced_shutdown=self.sessions[i].forced)
            folder=self.run_dir/f'source_{i+1:02d}'
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'display.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    def closeEvent(self,event):
        if self.sessions:
            self.closing=True;self.end_session();event.ignore()
        else:
            self.timer.stop();event.accept()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',help='Preselect source; never starts hardware automatically')
    args=parser.parse_args(argv)
    app=QApplication.instance() or QApplication(sys.argv[:1])
    window=MainWindow()
    if args.source:window.source.setCurrentText(args.source)
    window.show()
    return app.exec()


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
