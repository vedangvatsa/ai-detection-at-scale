#!/usr/bin/env python3
"""
Download pre-built data and model assets from GitHub Releases.

Usage:
    python scripts/download_assets.py           # Download everything
    python scripts/download_assets.py --data     # Download data only
    python scripts/download_assets.py --models   # Download models only
"""
import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RELEASE = 'v1.0-data'
REPO = 'vedangvatsa/ai-detection-at-scale'

DATA_ASSETS = [
    'corpus_raw.parquet.part_a',
    'corpus_raw.parquet.part_b',
    'corpus_features.parquet',
    'human_academic.parquet',
    'human_academic2.parquet',
    'human_news.parquet',
    'human_social.parquet',
    'corpus_summary.csv',
]

MODEL_ASSETS = [
    'register_classifier.joblib',
    'detector_all.joblib',
    'detector_academic.joblib',
    'detector_news.joblib',
    'detector_social.joblib',
    'detector_creative.joblib',
    'homoglyph_normalizer.joblib',
    'manifest.json',
]


def download_asset(asset, target_dir):
    target_path = os.path.join(target_dir, asset)
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / 1e6
        print(f"  SKIP {asset} (already exists, {size_mb:.0f} MB)")
        return
    os.makedirs(target_dir, exist_ok=True)
    url = f"https://github.com/{REPO}/releases/download/{RELEASE}/{asset}"
    print(f"  Downloading {asset}...")
    result = subprocess.run(['curl', '-L', '-o', target_path, url],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR downloading {asset}: {result.stderr}")
        return False
    size_mb = os.path.getsize(target_path) / 1e6
    print(f"  OK {asset} ({size_mb:.0f} MB)")
    return True


def reassemble_corpus():
    parts = [os.path.join(DATA_DIR, 'corpus_raw.parquet.part_a'),
             os.path.join(DATA_DIR, 'corpus_raw.parquet.part_b')]
    target = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if os.path.exists(target):
        print("  SKIP reassembly (corpus_raw.parquet already exists)")
        return
    if not all(os.path.exists(p) for p in parts):
        print("  SKIP reassembly (parts not downloaded)")
        return
    print("  Reassembling corpus_raw.parquet from parts...")
    with open(target, 'wb') as out:
        for part in parts:
            with open(part, 'rb') as f:
                out.write(f.read())
    size_gb = os.path.getsize(target) / 1e9
    print(f"  OK corpus_raw.parquet ({size_gb:.1f} GB)")
    os.remove(parts[0])
    os.remove(parts[1])
    print("  Cleaned up part files")


def main():
    parser = argparse.ArgumentParser(description='Download data and models from GitHub Releases.')
    parser.add_argument('--data', action='store_true', help='Download data files only')
    parser.add_argument('--models', action='store_true', help='Download model files only')
    args = parser.parse_args()

    download_all = not args.data and not args.models

    if download_all or args.data:
        print(f"\n=== Downloading data assets ({RELEASE}) ===")
        for asset in DATA_ASSETS:
            download_asset(asset, DATA_DIR)
        reassemble_corpus()

    if download_all or args.models:
        print(f"\n=== Downloading model assets ({RELEASE}) ===")
        for asset in MODEL_ASSETS:
            download_asset(asset, MODELS_DIR)

    print("\nDone. All assets downloaded.")


if __name__ == '__main__':
    main()
