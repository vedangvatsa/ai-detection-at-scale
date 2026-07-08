#!/usr/bin/env python3
"""
Generate a SHA256 checksums manifest for downloaded data and model assets.

Usage:
    python scripts/generate_checksums.py --out checksums.json
    python scripts/download_assets.py --checksums checksums.json --verify-only
"""
import os
import sys
import json
import hashlib
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Assets that download_assets.py expects; adjust if the release changes.
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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _collect_checksums(assets, directory):
    checksums = {}
    for asset in assets:
        path = os.path.join(directory, asset)
        if os.path.exists(path):
            checksums[asset] = _sha256_file(path)
        else:
            print(f"WARNING: missing {asset}", file=sys.stderr)
    return checksums


def main():
    parser = argparse.ArgumentParser(description='Generate SHA256 checksums manifest for release assets.')
    parser.add_argument('--out', type=str, default=os.path.join(PROJECT_DIR, 'checksums.json'),
                        help='Output JSON file path')
    args = parser.parse_args()

    print('Computing checksums...')
    manifest = {
        'data': _collect_checksums(DATA_ASSETS, DATA_DIR),
        'models': _collect_checksums(MODEL_ASSETS, MODELS_DIR),
    }

    # Flatten into a single asset-name -> checksum map for download_assets.py compatibility.
    flat = {}
    flat.update(manifest['data'])
    flat.update(manifest['models'])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(flat, f, indent=2, sort_keys=True)

    print(f'Wrote {len(flat)} checksums to {args.out}')


if __name__ == '__main__':
    main()
