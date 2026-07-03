#!/usr/bin/env python3
"""
Report TPR (recall) at 0.01% and 0.1% FPR for each register and all-register.

These are the metrics that matter for real-world deployment where false
positives on human text are extremely costly. Standard AUC and precision-at-
90%-recall do not capture performance in this ultra-low-FPR regime.

Outputs:
  results/tpr_at_low_fpr.csv
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

FPR_TARGETS = [0.0001, 0.001, 0.005, 0.01, 0.05]


def compute_tpr_at_fpr(y_true, y_scores, fpr_targets):
    """Compute TPR at specific FPR thresholds using ROC curve interpolation."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    results = {}
    for target in fpr_targets:
        idx = np.searchsorted(fpr, target, side='right')
        if idx < len(tpr):
            results[f'tpr_at_fpr_{target}'] = float(tpr[idx])
            results[f'threshold_at_fpr_{target}'] = float(thresholds[idx]) if idx < len(thresholds) else float('nan')
        else:
            results[f'tpr_at_fpr_{target}'] = float(tpr[-1])
            results[f'threshold_at_fpr_{target}'] = float(thresholds[-1])
    return results


def main():
    # Load manifest
    with open(os.path.join(MODELS_DIR, 'manifest.json')) as f:
        manifest = json.load(f)

    # Load feature data
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found. Run 02_extract_features.py first.")
        return

    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} texts from corpus_features.parquet")

    # Load all-register detector
    _all = joblib.load(os.path.join(MODELS_DIR, manifest['all_register_detector']))
    all_detector = _all['model'] if isinstance(_all, dict) else _all

    # Load per-register detectors
    detectors = {}
    for reg, fname in manifest['detectors'].items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            _d = joblib.load(path)
            detectors[reg] = _d['model'] if isinstance(_d, dict) else _d

    rows = []

    # Per-register evaluation
    for register in manifest['registers']:
        if register not in detectors:
            continue
        df_reg = df[df['register'] == register].dropna(subset=FEATURE_COLS)
        if len(df_reg) < 100:
            continue

        # Subsample for speed
        max_n = 100000
        if len(df_reg) > max_n:
            df_reg = df_reg.sample(max_n, random_state=42)

        X = df_reg[FEATURE_COLS].values
        y = df_reg['label'].values

        detector = detectors[register]
        y_proba = detector.predict_proba(X)[:, 1]  # detectors[reg] is already unwrapped

        result = compute_tpr_at_fpr(y, y_proba, FPR_TARGETS)
        result['register'] = register
        result['n_texts'] = len(df_reg)
        result['n_ai'] = int((y == 1).sum())
        result['n_human'] = int((y == 0).sum())
        rows.append(result)
        print(f"  {register}: TPR@0.01%FPR={result['tpr_at_fpr_0.0001']:.4f}, "
              f"TPR@0.1%FPR={result['tpr_at_fpr_0.001']:.4f}")

    # All-register evaluation
    print("\nEvaluating all-register detector...")
    df_all = df.dropna(subset=FEATURE_COLS)
    max_n = 200000
    if len(df_all) > max_n:
        df_all = df_all.sample(max_n, random_state=42)

    X_all = df_all[FEATURE_COLS].values
    y_all = df_all['label'].values
    y_proba_all = all_detector.predict_proba(X_all)[:, 1]

    result = compute_tpr_at_fpr(y_all, y_proba_all, FPR_TARGETS)
    result['register'] = 'all'
    result['n_texts'] = len(df_all)
    result['n_ai'] = int((y_all == 1).sum())
    result['n_human'] = int((y_all == 0).sum())
    rows.append(result)
    print(f"  all: TPR@0.01%FPR={result['tpr_at_fpr_0.0001']:.4f}, "
          f"TPR@0.1%FPR={result['tpr_at_fpr_0.001']:.4f}")

    # Save
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'tpr_at_low_fpr.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Register':<12} {'TPR@0.01%FPR':>14} {'TPR@0.1%FPR':>13} {'TPR@0.5%FPR':>13} {'TPR@1%FPR':>11} {'TPR@5%FPR':>11}")
    print("-" * 80)
    for _, row in out_df.iterrows():
        print(f"{row['register']:<12} "
              f"{row['tpr_at_fpr_0.0001']:>14.4f} "
              f"{row['tpr_at_fpr_0.001']:>13.4f} "
              f"{row['tpr_at_fpr_0.005']:>13.4f} "
              f"{row['tpr_at_fpr_0.01']:>11.4f} "
              f"{row['tpr_at_fpr_0.05']:>11.4f}")


if __name__ == '__main__':
    main()
