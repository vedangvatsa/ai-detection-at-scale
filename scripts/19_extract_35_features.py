#!/usr/bin/env python3
"""
Extract 35 stylometric features (11 original + 20 extended + 4 vocabulary richness).
Replaces the 31-feature extraction with the expanded feature set.
"""
import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_features, normalize_unicode

RANDOM_SEED = 42

def extract_row_features(row):
    """Extract features from a single row."""
    text = row['text']
    if not isinstance(text, str) or len(text.strip()) < 20:
        return None
    
    feats = extract_features(text, extended=True)
    if feats is None:
        return None
    
    return feats

def main():
    print("Loading raw corpus...")
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found.")
        return
    
    # Process in chunks to avoid OOM
    chunk_size = 50000
    output_path = os.path.join(DATA_DIR, 'corpus_features_35.parquet')
    
    # Read total count first
    df_sample = pd.read_parquet(raw_path, columns=['text']).head(1)
    total_rows = len(pd.read_parquet(raw_path, columns=['text']))
    print(f"Total texts: {total_rows}")
    
    all_features = []
    for i in range(0, total_rows, chunk_size):
        print(f"Processing chunk {i//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size}...")
        chunk = pd.read_parquet(raw_path, columns=['text']).iloc[i:i+chunk_size]
        
        # Extract features in parallel
        n_workers = min(cpu_count(), 4)
        with Pool(n_workers) as pool:
            results = pool.map(extract_row_features, [row for _, row in chunk.iterrows()])
        
        # Filter None results
        valid_results = [r for r in results if r is not None]
        all_features.extend(valid_results)
        print(f"  Valid: {len(valid_results)}/{len(chunk)}")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    feat_df = pd.DataFrame(all_features)
    
    # Save
    feat_df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Total valid: {len(feat_df)}")
    print(f"Features: {list(feat_df.columns)}")

if __name__ == '__main__':
    main()
