#!/usr/bin/env python3
"""
Multi-signal ensemble: combine stylometric score with neural detector score.

This script trains a meta-classifier that combines:
1. Stylometric RF probability (11 features)
2. Binoculars score (if available from 13_binoculars_baseline.py)
3. Optional: 31-feature stylometric probability

The meta-classifier (logistic regression) learns optimal weights for combining
the signals. Cross-validated AUC is compared to each signal alone.

If Binoculars scores are not available, the script trains a stacking ensemble
using the per-register stylometric detectors as base learners.

Outputs:
  results/ensemble_meta_results.csv
  models/ensemble_meta.joblib
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, normalize_unicode, ALL_FEATURE_COLS

ORIGINAL_FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

RANDOM_SEED = 42


def main():
    # Load models
    with open(os.path.join(MODELS_DIR, 'manifest.json')) as f:
        manifest = json.load(f)

    _all = joblib.load(os.path.join(MODELS_DIR, manifest['all_register_detector']))
    all_detector = _all['model'] if isinstance(_all, dict) else _all
    detectors = {}
    for reg, fname in manifest['detectors'].items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            _d = joblib.load(path)
            detectors[reg] = _d['model'] if isinstance(_d, dict) else _d

    # Load 31-feature model if available
    model_31_path = os.path.join(MODELS_DIR, 'detector_all_31.joblib')
    detector_31 = joblib.load(model_31_path) if os.path.exists(model_31_path) else None

    # Load feature data
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} texts")

    # Sample for training
    max_per_group = 50000
    parts = []
    for (reg, lab), grp in df.groupby(['register', 'label']):
        if len(grp) > max_per_group:
            parts.append(grp.sample(max_per_group, random_state=RANDOM_SEED))
        else:
            parts.append(grp)
    df_sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)
    df_sample = df_sample.dropna(subset=ORIGINAL_FEATURE_COLS)

    print(f"Sampled {len(df_sample)} texts for ensemble training")

    # ── Compute base learner scores ────────────────────────────────────────
    X_orig = df_sample[ORIGINAL_FEATURE_COLS].values
    y = df_sample['label'].values

    print("\nComputing base learner scores...")

    # Signal 1: All-register stylometric
    stylo_all_proba = all_detector.predict_proba(X_orig)[:, 1]
    stylo_all_auc = roc_auc_score(y, stylo_all_proba)
    print(f"  Stylometric (all-register): AUC={stylo_all_auc:.4f}")

    # Signal 2: Per-register stylometric (register-aware ensemble)
    stylo_reg_proba = np.zeros(len(df_sample))
    for i, (_, row) in enumerate(df_sample.iterrows()):
        reg = row['register']
        if reg in detectors:
            proba = detectors[reg].predict_proba(X_orig[i:i+1])[0, 1]
        else:
            proba = stylo_all_proba[i]
        stylo_reg_proba[i] = proba
    stylo_reg_auc = roc_auc_score(y, stylo_reg_proba)
    print(f"  Stylometric (register-aware): AUC={stylo_reg_auc:.4f}")

    # Signal 3: 31-feature stylometric (if available)
    stylo_31_proba = None
    if detector_31 is not None:
        feat_31_path = os.path.join(DATA_DIR, 'corpus_features_31.parquet')
        if os.path.exists(feat_31_path):
            df_31 = pd.read_parquet(feat_31_path)
            df_31_sample = df_31[df_31.index.isin(df_sample.index)]
            if len(df_31_sample) == len(df_sample) and all(c in df_31_sample.columns for c in ALL_FEATURE_COLS):
                X_31 = df_31_sample[ALL_FEATURE_COLS].values
                stylo_31_proba = detector_31.predict_proba(X_31)[:, 1]
                stylo_31_auc = roc_auc_score(y, stylo_31_proba)
                print(f"  Stylometric (31-feature): AUC={stylo_31_auc:.4f}")

    # Signal 4: Binoculars (if available)
    bino_scores = None
    bino_path = os.path.join(RESULTS_DIR, 'binoculars_scores.csv')
    if os.path.exists(bino_path):
        bino_df = pd.read_csv(bino_path)
        if 'binoculars_score_neg' in bino_df.columns and len(bino_df) == len(df_sample):
            bino_scores = bino_df['binoculars_score_neg'].values
            bino_auc = roc_auc_score(y, bino_scores)
            print(f"  Binoculars: AUC={bino_auc:.4f}")

    # ── Build meta-features ────────────────────────────────────────────────
    meta_features = [stylo_all_proba, stylo_reg_proba]
    meta_names = ['stylo_all', 'stylo_reg']

    if stylo_31_proba is not None:
        meta_features.append(stylo_31_proba)
        meta_names.append('stylo_31')

    if bino_scores is not None:
        meta_features.append(bino_scores)
        meta_names.append('binoculars')

    X_meta = np.column_stack(meta_features)

    # ── Train meta-classifier ──────────────────────────────────────────────
    print(f"\nTraining meta-classifier on {len(meta_names)} signals: {meta_names}")

    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    meta_clf = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    meta_clf.fit(X_meta_scaled, y)

    # Cross-validated evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    meta_aucs = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_meta_scaled, y)):
        fold_clf = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
        fold_scaler = StandardScaler()
        X_train = fold_scaler.fit_transform(X_meta_scaled[train_idx])
        X_test = fold_scaler.transform(X_meta_scaled[test_idx])
        fold_clf.fit(X_train, y[train_idx])
        y_proba = fold_clf.predict_proba(X_test)[:, 1]
        meta_aucs.append(roc_auc_score(y[test_idx], y_proba))

    meta_auc = np.mean(meta_aucs)
    meta_auc_std = np.std(meta_aucs)
    print(f"  Meta-ensemble AUC: {meta_auc:.4f} (+/- {meta_auc_std:.4f})")

    # Print meta-classifier coefficients
    print(f"\n  Meta-classifier coefficients:")
    for name, coef in zip(meta_names, meta_clf.coef_[0]):
        print(f"    {name}: {coef:.4f}")

    # ── Results table ──────────────────────────────────────────────────────
    results = []

    results.append({'method': 'stylo_all', 'auc': stylo_all_auc, 'n_signals': 1})
    results.append({'method': 'stylo_reg', 'auc': stylo_reg_auc, 'n_signals': 1})

    if stylo_31_proba is not None:
        results.append({'method': 'stylo_31', 'auc': stylo_31_auc, 'n_signals': 1})

    if bino_scores is not None:
        results.append({'method': 'binoculars', 'auc': bino_auc, 'n_signals': 1})

    results.append({'method': 'meta_ensemble', 'auc': meta_auc, 'n_signals': len(meta_names),
                    'auc_std': meta_auc_std})

    # Save
    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, 'ensemble_meta_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Save meta-classifier
    meta_model = {'classifier': meta_clf, 'scaler': scaler, 'signal_names': meta_names}
    joblib.dump(meta_model, os.path.join(MODELS_DIR, 'ensemble_meta.joblib'))
    print(f"Saved meta-classifier to {MODELS_DIR}/ensemble_meta.joblib")

    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'AUC':>8} {'Signals':>8}")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"{row['method']:<25} {row['auc']:>8.4f} {row['n_signals']:>8}")


if __name__ == '__main__':
    main()
