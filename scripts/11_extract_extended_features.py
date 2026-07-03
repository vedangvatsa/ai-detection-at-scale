#!/usr/bin/env python3
"""
Extract all 31 features (original 11 + extended 20) from corpus_raw.parquet.
Uses multiprocessing for parallel extraction. Trains and compares to 11-feature baseline.

Outputs:
  data/corpus_features_31.parquet
  results/extended_feature_comparison.csv
  results/extended_feature_importance.csv
  models/detector_all_31.joblib
"""
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import (
    extract_features, ALL_FEATURE_COLS, ORIGINAL_FEATURE_COLS,
)

RANDOM_SEED = 42
N_WORKERS = max(1, cpu_count() - 1)


def _extract_one(args):
    """Worker function: extract 31 features from a single text."""
    text, label, register, model, source = args
    feats = extract_features(text, extended=True)
    if feats is None:
        return None
    feats['label'] = label
    feats['register'] = register
    feats['model'] = model
    feats['source'] = source
    return feats


def main():
    # Load raw corpus
    in_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 01_fetch_data.py first.")
        return

    df = pd.read_parquet(in_path, columns=['text', 'label', 'register', 'model', 'source'])
    print(f"Loaded {len(df)} texts from corpus_raw.parquet")

    # Extract 31 features with multiprocessing
    print(f"Extracting 31 features per text using {N_WORKERS} workers...")
    args_list = list(zip(
        df['text'].values,
        df['label'].values,
        df['register'].values,
        df.get('model', pd.Series([''] * len(df))).values,
        df.get('source', pd.Series([''] * len(df))).values,
    ))

    feature_rows = []
    failed = 0
    with Pool(N_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(_extract_one, args_list, chunksize=500),
                           total=len(args_list), desc="Extracting 31 features"):
            if result is None:
                failed += 1
            else:
                feature_rows.append(result)

    print(f"  Extracted: {len(feature_rows)}, Failed: {failed}")
    feat_df = pd.DataFrame(feature_rows)
    feat_df = feat_df.dropna(subset=['mtld'])

    out_path = os.path.join(DATA_DIR, 'corpus_features_31.parquet')
    feat_df.to_parquet(out_path, index=False)
    print(f"Saved {len(feat_df)} rows to {out_path}")

    # Train and compare
    print("\n=== Training 31-feature classifier ===")
    max_per_group = 50000
    parts = []
    for (reg, lab), grp in feat_df.groupby(['register', 'label']):
        if len(grp) > max_per_group:
            parts.append(grp.sample(max_per_group, random_state=RANDOM_SEED))
        else:
            parts.append(grp)
    df_sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)

    X_31 = df_sample[ALL_FEATURE_COLS].values
    X_11 = df_sample[ORIGINAL_FEATURE_COLS].values
    y = df_sample['label'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    results = []

    for label, X, cols in [("11-feature", X_11, ORIGINAL_FEATURE_COLS),
                           ("31-feature", X_31, ALL_FEATURE_COLS)]:
        print(f"\n  {label} classifier (5-fold CV)...")
        aucs, accs, f1s = [], [], []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            y_pred = rf.predict(X[test_idx])
            y_proba = rf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], y_proba))
            accs.append(accuracy_score(y[test_idx], y_pred))
            f1s.append(f1_score(y[test_idx], y_pred))
            print(f"    Fold {fold+1}: AUC={aucs[-1]:.4f}")

        results.append({
            'feature_set': label,
            'n_features': len(cols),
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'accuracy_mean': np.mean(accs),
            'f1_mean': np.mean(f1s),
        })
        print(f"  {label}: AUC={np.mean(aucs):.4f} (+/-{np.std(aucs):.4f})")

    comp_df = pd.DataFrame(results)
    comp_path = os.path.join(RESULTS_DIR, 'extended_feature_comparison.csv')
    comp_df.to_csv(comp_path, index=False)
    print(f"\nSaved comparison to {comp_path}")

    # Train final 31-feature model on all data
    print("\n=== Training final 31-feature model ===")
    rf_final = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_final.fit(X_31, y)
    model_path = os.path.join(MODELS_DIR, 'detector_all_31.joblib')
    joblib.dump(rf_final, model_path)
    print(f"Saved to {model_path}")

    # Permutation importance
    print("\n=== Computing permutation importance ===")
    X_test = X_31
    y_test = y
    perm_imp = permutation_importance(rf_final, X_test, y_test, n_repeats=10,
                                       scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1)
    imp_rows = []
    for i, col in enumerate(ALL_FEATURE_COLS):
        imp_rows.append({
            'feature': col,
            'importance_mean': perm_imp.importances_mean[i],
            'importance_std': perm_imp.importances_std[i],
        })
    imp_df = pd.DataFrame(imp_rows).sort_values('importance_mean', ascending=False)
    imp_path = os.path.join(RESULTS_DIR, 'extended_feature_importance.csv')
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved to {imp_path}")

    print("\nTop 10 features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))

    # Per-register evaluation
    print("\n=== Per-register 31-feature AUC ===")
    for register in ['academic', 'news', 'social', 'creative']:
        df_reg = df_sample[df_sample['register'] == register]
        if len(df_reg) < 100:
            continue
        X_reg = df_reg[ALL_FEATURE_COLS].values
        y_reg = df_reg['label'].values
        y_proba = rf_final.predict_proba(X_reg)[:, 1]
        auc = roc_auc_score(y_reg, y_proba)
        print(f"  {register}: AUC={auc:.4f} (n={len(df_reg)})")

    print("\nDone.")


if __name__ == '__main__':
    main()
