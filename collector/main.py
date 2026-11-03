#!/usr/bin/env python3
"""PoseForge Studio: python main.py --data ./Real_Posture_Dataset"""
import argparse
from pathlib import Path
from posture_collector.storage import DEFAULT_COLLECTOR_DATA_ROOT

def main():
    parser=argparse.ArgumentParser(description='PoseForge multi-camera posture recording GUI')
    parser.add_argument('--data', default=str(DEFAULT_COLLECTOR_DATA_ROOT))
    args=parser.parse_args()
    old_data = Path(__file__).resolve().parent / 'Real_Posture_Dataset'
    if Path(args.data).expanduser().resolve() == DEFAULT_COLLECTOR_DATA_ROOT and old_data.exists():
        raise SystemExit('Old collector/Real_Posture_Dataset still exists. '
                         'Use --data PATH to open an existing dataset, or migrate the old '
                         'runtime directory before running with defaults.')
    try:from posture_collector.ui import launch
    except ImportError as e:
        raise SystemExit('Thiếu PySide6 hoặc OpenCV. Cài: pip install -r requirements.txt\n'+str(e))
    return launch(args.data)

if __name__=='__main__':raise SystemExit(main())
