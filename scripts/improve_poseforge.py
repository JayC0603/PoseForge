#!/usr/bin/env python3
"""Development-only geometry/architecture improvement; no test evaluation stage."""
import argparse
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from shared import model_improvement as mi
    from shared import real_training as rt
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',required=True,choices=['prepare','screen','confirm','cv','cv-robustness','finalize','deploy','report'])
    ap.add_argument('--source-run',type=Path,default=ROOT/'artifacts/training/real_v001')
    ap.add_argument('--raw-data',type=Path,default=ROOT/'artifacts/dataset_working/Real_Posture_Dataset')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/improvement')
    ap.add_argument('--baseline',type=Path,default=ROOT/'artifacts/models/mlp_wide/v001')
    ap.add_argument('--run',type=Path)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    logs=(args.run if args.run else args.output)/'logs';logs.mkdir(parents=True,exist_ok=True)
    filename=logs/f'{args.stage}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")}.log'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(filename,encoding='utf-8'),logging.StreamHandler()])
    logging.info('Command: %s',' '.join(sys.argv))
    mi.torch.set_num_threads(2)
    try:
        if args.stage=='prepare':
            result=str(mi.prepare(args.source_run,args.output,args.raw_data,args.baseline))
        else:
            if args.run is None:ap.error('--run required after prepare')
            if args.device.startswith('cuda') and not mi.torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            if args.stage=='screen':result=mi.screen(args.run,args.device,args.resume)
            elif args.stage=='confirm':result=mi.confirm(args.run,args.device,args.resume)
            elif args.stage=='cv':result=mi.cross_validate(args.run,args.device,args.resume)
            elif args.stage=='cv-robustness':result=mi.cross_validate_robustness(args.run,args.device,args.resume)
            elif args.stage=='finalize':result=mi.finalize(args.run,args.device,args.resume)
            elif args.stage=='deploy':result=mi.deploy(args.run)
            else:result=mi.report(args.run)
        logging.info('Completed %s: %s',args.stage,result)
    except BaseException:
        logging.exception('Improvement stage failed/interrupted; raw data and baseline are preserved.')
        raise


if __name__=='__main__':main()
