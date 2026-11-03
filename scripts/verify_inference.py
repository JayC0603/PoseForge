"""Read-only deployment checks and development-video desktop smoke (never a webcam)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
AUDIT=ROOT/'artifacts/inference_runs/implementation_audit'
DEV_VIDEO=ROOT/'artifacts/dataset_working/Real_Posture_Dataset/recordings/DGD/session_20261027_1124/label_0/take_001/cam01_raw.mp4'


def write(name,value):
    AUDIT.mkdir(parents=True,exist_ok=True)
    (AUDIT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(value,ensure_ascii=False))


def preservation():
    from inference.models import sha256
    before=json.loads((AUDIT/'preservation_before.json').read_text(encoding='utf-8'))
    changed=[];missing=[]
    for name,record in before.items():
        path=ROOT/name
        if not path.is_file():missing.append(name)
        elif sha256(path)!=record['sha256']:changed.append(name)
    additions=[]
    for top in ['artifacts','collector','shared','scripts','notebooks','configs','reports','docs','tests']:
        for path in (ROOT/top).rglob('*'):
            rel=path.relative_to(ROOT).as_posix()
            if path.is_file() and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts and not rel.startswith('artifacts/inference_runs/') and rel not in before:
                additions.append(rel)
    result=dict(original_files=len(before),changed=changed,missing=missing,added=sorted(additions),
                content_preserved=not changed and not missing)
    write('preservation_after.json',result)
    if changed or missing:raise SystemExit('Preservation check failed')


def bundles():
    import numpy as np
    import torch
    import cv2
    import PySide6
    import onnxruntime as ort
    from inference.models import discover_models, provenance
    from inference.pipeline import PostureClassifier
    entries,errors=discover_models()
    results=[]
    for entry in entries:
        # Load sequentially; never retain every backend/model simultaneously.
        rng=np.random.default_rng(42)
        length=int(entry.name.split('–')[0].split('-')[-1].strip()) if entry.research else 1
        samples=rng.normal(size=(32,length,52) if entry.research else (32,13,4)).astype(np.float32)
        predictions={};latencies={}
        for backend in entry.backends:
            classifier=PostureClassifier(entry.path,backend,'cpu')
            predictions[backend]=np.stack([classifier.predict(x) for x in samples])
            measured=[]
            for _ in range(100):
                start=time.perf_counter();classifier.predict(samples[0]);measured.append((time.perf_counter()-start)*1000)
            latencies[backend]=dict(median_ms=float(np.median(measured)),p95_ms=float(np.percentile(measured,95)))
            del classifier
        difference=float(np.max(np.abs(predictions['pt']-predictions['onnx']))) if len(predictions)==2 else None
        if difference is not None:np.testing.assert_allclose(predictions['pt'],predictions['onnx'],atol=1e-5,rtol=1e-5)
        results.append(dict(name=entry.name,backends=entry.backends,labels=entry.labels,feature_dim=52,output_dim=6,
                            sequence_length=length,finite=all(np.isfinite(a).all() for a in predictions.values()),
                            max_probability_difference=difference,classifier_timings=latencies,provenance=provenance(entry.path)))
    write('model_verification.json',dict(models=results,errors=errors,python=sys.executable,torch=torch.__version__,
        cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0),
        opencv=cv2.__version__,pyside=PySide6.__version__,onnxruntime=ort.__version__,providers=ort.get_available_providers()))
    if len(entries)!=4 or errors:raise SystemExit('Model catalog not fully available')


def offline_outputs():
    import cv2
    import numpy as np
    onnx=sorted((ROOT/'artifacts/inference_runs').glob('desktop_offline_smoke_v*'))[-1]
    pt=sorted((ROOT/'artifacts/inference_runs').glob('desktop_offline_pt_v*'))[-1]
    rows={}
    for backend,path in [('onnx',onnx),('pt',pt)]:
        rows[backend]=[json.loads(line) for line in (path/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows[backend])==71
        assert all(r['feature_dim']==52 and len(r['probs'])==6 and np.isfinite(r['probs']).all() for r in rows[backend])
    a=np.array([r['probs'] for r in rows['onnx']]);b=np.array([r['probs'] for r in rows['pt']])
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)
    assert [r['class_id'] for r in rows['onnx']]==[r['class_id'] for r in rows['pt']]
    cap=cv2.VideoCapture(str(onnx/'annotated.mp4'));decoded=0;shape=None
    try:
        assert cap.isOpened()
        while True:
            ok,frame=cap.read()
            if not ok:break
            decoded+=1;shape=frame.shape
    finally:cap.release()
    assert decoded==71 and not (pt/'annotated.mp4').exists()
    write('offline_verification.json',dict(onnx_run=str(onnx),pt_run=str(pt),frames=71,
        predicted_frames=71,finite=True,feature_dim=52,output_dim=6,
        max_probability_difference=float(np.abs(a-b).max()),same_class_ids=True,
        decoded_annotated_frames=decoded,annotated_shape=shape,pt_recording_disabled=True,
        accuracy_claim=False))


def gui_smoke(sources,seconds,visible):
    if not visible:os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from inference.gui import MainWindow
    if not DEV_VIDEO.is_file():raise FileNotFoundError(DEV_VIDEO)
    app=QApplication.instance() or QApplication([])
    window=MainWindow();window.source.setCurrentText(str(DEV_VIDEO));window.extra_sources=[str(DEV_VIDEO)]*(sources-1)
    window.options['export_predictions']=True
    window.options['export_video']=True
    window.show();window.raise_();window.activateWindow();app.processEvents();window.start_session()
    first=None;deadline=time.monotonic()+90;timed_out=False;screenshot_saved=False
    while window.sessions:
        app.processEvents()
        if len(window.panels)==sources and all(p.preview.display_rate.total>=2 for p in window.panels) and first is None:
            first=time.monotonic()
            window.grab().save(str(AUDIT/f'gui_{sources}_sources.png'))
            screenshot_saved=True
        if not window.stopping and first is not None and time.monotonic()-first>=seconds:
            window.grab().save(str(AUDIT/f'gui_{sources}_sources.png'))
            window.end_session()
        if time.monotonic()>deadline:
            timed_out=True;window.end_session()
        time.sleep(.005)
    run_dir=window.run_dir
    visibility=dict(visible=window.isVisible(),minimized=window.isMinimized(),active=window.isActiveWindow(),
                    exposed=window.windowHandle().isExposed())
    window.close();app.processEvents()
    reports=[]
    for folder in sorted(run_dir.glob('source_*')):
        performance=json.loads((folder/'performance.json').read_text()) if (folder/'performance.json').exists() else {}
        display=json.loads((folder/'display.json').read_text())
        reports.append(dict(source=folder.name,performance=performance,display=display))
    result=dict(source=str(DEV_VIDEO.relative_to(ROOT)),role='existing development material; functional smoke, not independent accuracy',
                source_count=sources,visible_window=visible,qt_platform=app.platformName(),run_dir=str(run_dir),
                window_visibility=visibility,
                requested_seconds_after_first_paints=seconds,ended_at_video_eof=first is not None and time.monotonic()-first<seconds,
                screenshot_saved=screenshot_saved,timed_out=timed_out,reports=reports)
    write(f'gui_smoke_{sources}.json',result)
    assert not timed_out and len(reports)==sources
    assert all(r['display']['painted_frames']>2 and not r['display']['forced_shutdown'] for r in reports)
    assert all(r['performance'].get('processed_frames',0)>2 and r['performance']['state']!='ERROR' for r in reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',action='store_true')
    parser.add_argument('--offline-check',action='store_true')
    parser.add_argument('--preservation',action='store_true')
    parser.add_argument('--gui-smoke',action='store_true')
    parser.add_argument('--sources',type=int,choices=[1,2,3,4],default=1)
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--visible',action='store_true',help='Show real MP4 Qt window; never opens hardware')
    args=parser.parse_args()
    if args.models:bundles()
    if args.offline_check:offline_outputs()
    if args.gui_smoke:gui_smoke(args.sources,args.seconds,args.visible)
    if args.preservation:preservation()


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
