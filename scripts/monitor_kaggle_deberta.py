#!/usr/bin/env python3
"""Monitor status of the Kaggle DeBERTa-v3-large fine-tuning kernel."""
import os
import sys
import time
import subprocess
import datetime

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
KAGGLE_BIN = os.path.join(PROJECT_DIR, '.venv', 'bin', 'kaggle')
KERNEL = 'vedangvatsa123/turingbench-deberta-v3-large-fine-tune'
LOG_DIR = os.path.join(PROJECT_DIR, 'kaggle_output', 'v6')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'monitor.log')


def main():
    print(f'Starting monitor for {KERNEL}')
    print(f'Log: {LOG_PATH}')
    with open(LOG_PATH, 'a') as f:
        f.write(f"\n--- Monitor started {datetime.datetime.now()} ---\n")
        while True:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                res = subprocess.run(
                    [KAGGLE_BIN, 'kernels', 'status', '-k', KERNEL],
                    capture_output=True, text=True, timeout=60
                )
                status = res.stdout.strip().replace('\n', ' ')
                line = f'{now} {status}'
                print(line)
                f.write(line + '\n')
                f.flush()
                if any(s in status for s in ('COMPLETE', 'FAILED', 'ERROR')):
                    f.write(f'{now} Terminal status reached. Exiting.\n')
                    break
            except Exception as e:
                line = f'{now} ERROR: {e}'
                print(line)
                f.write(line + '\n')
                f.flush()
            time.sleep(120)


if __name__ == '__main__':
    main()
