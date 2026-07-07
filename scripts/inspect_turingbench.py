#!/usr/bin/env python3
"""Inspect TuringBench dataset structure and size."""
import os
import sys
from datasets import load_dataset
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

def main():
    print("Loading TuringBench dataset...")
    try:
        tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Splits: {list(tb.keys())}")
    for split in tb.keys():
        ds = tb[split]
        print(f"\nSplit: {split}, Rows: {len(ds)}")
        print(f"Columns: {ds.column_names}")
        if 'label' in ds.column_names:
            labels = pd.Series(ds['label']).value_counts().to_dict()
            print(f"Label distribution: {labels}")
        # show first few labels
        print(f"First 10 labels: {ds['label'][:10]}")

if __name__ == '__main__':
    main()
