#!/usr/bin/env python3
"""Windows real-data workflow; all derived artifacts are versioned outside the input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from shared import real_training as rt
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["audit", "prepare", "baseline", "improve", "finalize", "export", "internal-test", "full-data", "report", "sequences", "gru16", "gru32", "compare", "verify"])
    ap.add_argument('--split-strategy',choices=['subject_wise','temporal_within_take','take_level'],default='subject_wise')
    ap.add_argument('--train-stride',type=int,choices=[4,8],default=4)
    ap.add_argument('--min-sequence-coverage',type=float,default=.8)
    ap.add_argument('--max-sequence-gap',type=float,default=.25)
    ap.add_argument('--max-missing-run',type=int,default=3)
    ap.add_argument('--per-source-window-cap',type=int,default=48)
    ap.add_argument('--split-candidates',type=int,default=512)
    ap.add_argument('--source-run',type=Path,default=ROOT/'artifacts/training/real_v001',help='Verified prior immutable features/eligibility for temporal/full-data workflow')
    ap.add_argument('--purge-seconds',type=float,default=.25,help='Symmetric capture-time exclusion radius on each side of a split boundary')
    ap.add_argument('--ratios',type=float,nargs=3,default=[.8,.1,.1])
    ap.add_argument('--min-eval-frames',type=int,default=8)
    ap.add_argument('--min-eval-seconds',type=float,default=.10)
    ap.add_argument('--min-eval-takes-per-class',type=int,default=2)
    ap.add_argument('--min-eval-subjects-per-class',type=int,default=2)
    ap.add_argument("--data", type=Path, default=ROOT / "artifacts/dataset_working/Real_Posture_Dataset")
    ap.add_argument("--reports", type=Path, default=ROOT / "artifacts/reports")
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/training")
    ap.add_argument("--run", type=Path, help="Prepared real_vNNN directory; required after prepare")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", type=Path, help="JSON overrides for baseline defaults")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--validation-subjects", nargs="+")
    ap.add_argument("--test-subjects", nargs="+")
    ap.add_argument("--resume", action="store_true", help="Resume trusted local optimizer/RNG checkpoints; completed experiments are skipped")
    args = ap.parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    log_dir = args.run / "logs" if args.run else args.reports / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{args.stage}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log, encoding="utf-8"), logging.StreamHandler()])
    logging.info("Command: %s", " ".join(sys.argv))
    try:
        if args.split_strategy=='take_level':
            from shared import take_gru_training as tg
            import torch
            torch.set_num_threads(2)
            if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            if args.stage=='audit':
                output=args.output if args.output!=ROOT/'artifacts/training' else ROOT/'artifacts/experiments/take_level_gru'
                options=dict(seed=args.seed,split_candidates=args.split_candidates,per_source_window_cap=args.per_source_window_cap,
                    sequence=dict(stride=args.train_stride,min_coverage=args.min_sequence_coverage,max_gap_seconds=args.max_sequence_gap,max_missing_run=args.max_missing_run))
                result=str(tg.audit_split(args.data,args.source_run,output,options))
            else:
                if args.run is None:ap.error('--run required after take-level audit')
                if args.stage in ['prepare','sequences']:result=tg.prepare_sequences(args.run)
                elif args.stage=='verify':result=tg.verify_run(args.run,sources=True)
                elif args.stage in ['baseline','gru16','gru32']:result=tg.train_model(args.run,'mlp_control' if args.stage=='baseline' else args.stage,args.device,args.resume)
                elif args.stage=='compare':result=tg.compare(args.run,args.device)
                elif args.stage=='internal-test':result=tg.internal_test(args.run,args.device)
                elif args.stage in ['finalize','full-data']:result=tg.finalize(args.run,args.device,args.resume)
                elif args.stage=='export':result=tg.deploy(args.run)
                elif args.stage=='report':result=tg.report(args.run)
                else:ap.error('Unsupported take-level stage')
            logging.info('Take-level stage %s complete: %s',args.stage,result)
            return
        if args.split_strategy=='temporal_within_take':
            from shared import temporal_training as tt
            import torch
            torch.set_num_threads(2)
            if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
            if args.stage=='audit':
                output=args.output if args.output!=ROOT/'artifacts/training' else ROOT/'artifacts/experiments/temporal_within_take_80_10_10'
                options=dict(ratios=args.ratios,purge_seconds=args.purge_seconds,clock_tolerance_ms=50.,
                    min_eval_frames=args.min_eval_frames,min_eval_seconds=args.min_eval_seconds,
                    min_eval_takes_per_class=args.min_eval_takes_per_class,min_eval_subjects_per_class=args.min_eval_subjects_per_class)
                result=str(tt.audit(args.data,args.source_run,output,options))
            else:
                if args.run is None:ap.error('--run required after temporal audit')
                if args.stage=='prepare':result=tt.freeze(args.run)
                elif args.stage=='baseline':result=tt.train_temporal(args.run,args.device,args.resume)
                elif args.stage=='internal-test':result=tt.internal_test(args.run,args.device)
                elif args.stage in ['full-data','finalize']:result=tt.fit_full_data(args.run,args.device,args.resume)
                elif args.stage=='export':result=tt.deploy(args.run)
                elif args.stage=='report':result=tt.report(args.run)
                else:ap.error('Unsupported temporal stage')
            logging.info('Temporal/full-data stage %s complete: %s',args.stage,result)
            return
        if args.stage in ['internal-test','full-data','report']:ap.error('This stage requires --split-strategy temporal_within_take')
        if args.stage == "audit":
            # Existing reports are evidence, never silently replaced by a rerun.
            if (args.reports / "training_preflight.json").exists():
                raise FileExistsError("Audit exists; use --reports with a new directory")
            result = rt.audit(args.data, args.reports, ROOT / "artifacts/reextract_backups/20261030T115911Z_cc9dbeb9/manifest.json")
        elif args.stage == "prepare":
            result = str(rt.prepare(args.data, args.reports, args.output, args.validation_subjects, args.test_subjects, args.seed))
        else:
            if args.run is None:
                ap.error("--run is required for training/finalize/export")
            if args.stage in {"baseline", "improve", "finalize"}:
                import torch
                torch.set_num_threads(2)
                if args.device.startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable")
                rt.write_json(args.run / "environment.json", dict(python=sys.version, torch=str(torch.__version__),
                    device=args.device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
            if args.stage == "baseline":
                cfg = rt.read_json(args.config) if args.config else {}
                result = rt.baselines(args.run, args.device, {**cfg, "seed":args.seed}, args.resume)
            elif args.stage == "improve":
                result = rt.improvements(args.run, args.device, args.resume)
            elif args.stage == "finalize":
                result = str(rt.finalize(args.run, ROOT / "artifacts/models", args.device, args.resume))
            else:
                result = rt.export_and_validate(args.run)
        logging.info("Stage %s complete: %s", args.stage, result)
    except BaseException:
        logging.exception("Stage %s interrupted/failed; source data untouched. Last completed epoch is resumable.", args.stage)
        raise


if __name__ == "__main__":
    main()
