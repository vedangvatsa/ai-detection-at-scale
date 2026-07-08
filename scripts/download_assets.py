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
import json
import hashlib
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RELEASE = 'v1.0-data'
REPO = 'vedangvatsa/ai-detection-at-scale'


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _remote_size(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return int(resp.headers.get('Content-Length', -1))
    except Exception:
        return -1

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


def download_asset(asset, target_dir, expected_checksums=None, retries=3):
    target_path = os.path.join(target_dir, asset)
    expected_sha = expected_checksums.get(asset) if expected_checksums else None

    if os.path.exists(target_path):
        actual_sha = _sha256_file(target_path)
        if expected_sha and actual_sha != expected_sha:
            print(f"  RE-DOWNLOAD {asset} (checksum mismatch)")
        else:
            size_mb = os.path.getsize(target_path) / 1e6
            print(f"  SKIP {asset} (already exists, {size_mb:.0f} MB)")
            return True

    os.makedirs(target_dir, exist_ok=True)
    url = f"https://github.com/{REPO}/releases/download/{RELEASE}/{asset}"

    expected_size = _remote_size(url)
    if expected_size == 0:
        print(f"  ERROR {asset}: remote file appears empty or release {RELEASE} not found")
        return False

    print(f"  Downloading {asset} (expected {expected_size / 1e6:.0f} MB)...")
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ['curl', '-fsSL', '--retry', '2', '-o', target_path, url],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            actual_size = os.path.getsize(target_path)
            if expected_size > 0 and actual_size != expected_size:
                print(f"  WARNING {asset}: size mismatch ({actual_size} vs {expected_size})")
                if attempt < retries:
                    continue
                print(f"  ERROR {asset}: download size mismatch after {retries} attempts")
                return False
            actual_sha = _sha256_file(target_path)
            if expected_sha and actual_sha != expected_sha:
                print(f"  WARNING {asset}: checksum mismatch (expected {expected_sha}, got {actual_sha})")
                if attempt < retries:
                    continue
                print(f"  ERROR {asset}: checksum mismatch after {retries} attempts")
                return False
            size_mb = actual_size / 1e6
            print(f"  OK {asset} ({size_mb:.0f} MB, sha256={actual_sha[:16]}...)")
            return True
        else:
            print(f"  ERROR downloading {asset} (attempt {attempt}/{retries}): {result.stderr.strip()}")
            if attempt < retries:
                continue
    return False


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
    parser.add_argument('--checksums', type=str, default=None,
                        help='JSON file mapping asset names to expected SHA256 checksums')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing files against checksums; do not download')
    args = parser.parse_args()

    expected_checksums = {}
    if args.checksums and os.path.exists(args.checksums):
        with open(args.checksums) as f:
            expected_checksums = json.load(f)
        print(f"Loaded {len(expected_checksums)} expected checksums from {args.checksums}")
    elif args.checksums:
        print(f"WARNING: checksums file {args.checksums} not found; skipping SHA256 verification")

    if args.verify_only:
        print("\n=== Verifying existing assets ===")
        all_ok = True
        targets = [(a, DATA_DIR) for a in DATA_ASSETS] + [(a, MODELS_DIR) for a in MODEL_ASSETS]
        for asset, target_dir in targets:
            target_path = os.path.join(target_dir, asset)
            if not os.path.exists(target_path):
                print(f"  MISSING {asset}")
                all_ok = False
                continue
            if expected_checksums.get(asset):
                actual_sha = _sha256_file(target_path)
                if actual_sha != expected_checksums[asset]:
                    print(f"  MISMATCH {asset}: expected {expected_checksums[asset]}, got {actual_sha}")
                    all_ok = False
                else:
                    print(f"  OK {asset} (sha256 verified)")
            else:
                print(f"  OK {asset} (no checksum provided)")
        sys.exit(0 if all_ok else 1)

    download_all = not args.data and not args.models
    failed = []

    if download_all or args.data:
        print(f"\n=== Downloading data assets ({RELEASE}) ===")
        for asset in DATA_ASSETS:
            if not download_asset(asset, DATA_DIR, expected_checksums):
                failed.append(asset)
        reassemble_corpus()

    if download_all or args.models:
        print(f"\n=== Downloading model assets ({RELEASE}) ===")
        for asset in MODEL_ASSETS:
            if not download_asset(asset, MODELS_DIR, expected_checksums):
                failed.append(asset)

    if failed:
        print(f"\nERROR: {len(failed)} asset(s) failed to download: {failed}")
        sys.exit(1)

    print("\nDone. All assets downloaded and verified.")


if __name__ == '__main__':
    main()
