#!/usr/bin/env python3
"""
Evaluate 35 features on a small sample (memory-efficient).
Compares 11-feature vs 35-feature performance.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_features, normalize_unicode, ORIGINAL_FEATURE_COLS, EXTENDED_FEATURE_COLS

RANDOM_SEED = 42
SAMPLE_SIZE = 10000  # Small sample for memory efficiency

def main():
    print("Loading raw corpus (small sample)...")
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found.")
        return
    
    # Load only needed columns and sample
    df = pd.read_parquet(raw_path, columns=['text', 'label', 'register'])
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
    print(f"Sampled {len(df)} texts")
    
    # Extract features
    print("Extracting 35 features...")
    features_11 = []
    features_35 = []
    labels = []
    registers = []
    
    for _, row in df.iterrows():
        text = row['text']
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        
        feats = extract_features(text, extended=True)
        if feats is None:
            continue
        
        # 11 features
        feats_11 = {k: feats[k] for k in ORIGINAL_FEATURE_COLS if k in feats}
        features_11.append(feats_11)
        
        # 35 features
        feats_35 = {k: feats[k] for k in ORIGINAL_FEATURE_COLS + EXTENDED_FEATURE_COLS if k in feats}
        features_35.append(feats_35)
        
        labels.append(row['label'])
        registers.append(row['register'])
    
    print(f"Valid extractions: {len(labels)}")
    
    # Convert to DataFrames
    df_11 = pd.DataFrame(features_11)
    df_35 = pd.DataFrame(features_35)
    y = np.array(labels)
    
    # Train/test split
    X_train_11, X_test_11, y_train, y_test = train_test_split(df_11, y, test_size=0.2, random_state=RANDOM_SEED)
    X_train_35, X_test_35, _, _ = train_test_split(df_35, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Train 11-feature model
    print("Training 11-feature model...")
    rf_11 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_11.fit(X_train_11, y_train)
    auc_11 = roc_auc_score(y_test, rf_11.predict_proba(X_test_11)[:, 1])
    print(f"11-feature AUC: {auc_11:.4f}")
    
    # Train 35-feature model
    print("Training 35-feature model...")
    rf_35 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf_35.fit(X_train_35, y_train)
    auc_35 = roc_auc_score(y_test, rf_35.predict_proba(X_test_35)[:, 1])
    print(f"35-feature AUC: {auc_35:.4f}")
    
    # Save results
    results = {
        'n_features_11': len(ORIGINAL_FEATURE_COLS),
        'n_features_35': len(ORIGINAL_FEATURE_COLS) + len(EXTENDED_FEATURE_COLS),
        'auc_11': auc_11,
        'auc_35': auc_35,
        'improvement': auc_35 - auc_11,
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(RESULTS_DIR, 'feature_35_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
