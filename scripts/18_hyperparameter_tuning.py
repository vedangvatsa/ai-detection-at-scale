#!/usr/bin/env python3
"""
Hyperparameter tuning for Random Forest models.
Uses randomized search to find better hyperparameters for per-register detectors.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, normalize_unicode

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

RANDOM_SEED = 42

def main():
    print("Loading feature corpus...")
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path, columns=['label', 'register'] + FEATURE_COLS)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"Loaded {len(df)} texts")

    # Hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }

    results = []

    for register in ['academic', 'news', 'social', 'creative']:
        print(f"\n=== Tuning {register} ===")
        df_reg = df[df['register'] == register]
        
        # Balance sample for tuning (faster)
        n_per_label = min(5000, len(df_reg[df_reg['label'] == 0]), len(df_reg[df_reg['label'] == 1]))
        df_bal = pd.concat([
            df_reg[df_reg['label'] == 0].sample(n_per_label, random_state=RANDOM_SEED),
            df_reg[df_reg['label'] == 1].sample(n_per_label, random_state=RANDOM_SEED)
        ])
        
        X = df_bal[FEATURE_COLS].values
        y = df_bal['label'].values
        
        # Baseline model
        rf_baseline = RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf_baseline.fit(X, y)
        baseline_auc = roc_auc_score(y, rf_baseline.predict_proba(X)[:, 1])
        print(f"Baseline AUC: {baseline_auc:.4f}")
        
        # Randomized search
        rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        search = RandomizedSearchCV(
            rf, param_dist, n_iter=20, cv=cv, scoring='roc_auc',
            random_state=RANDOM_SEED, n_jobs=-1, verbose=1
        )
        search.fit(X, y)
        
        best_auc = search.best_score_
        best_params = search.best_params_
        print(f"Best CV AUC: {best_auc:.4f}")
        print(f"Best params: {best_params}")
        
        results.append({
            'register': register,
            'baseline_auc': baseline_auc,
            'tuned_auc': best_auc,
            'improvement': best_auc - baseline_auc,
            'best_params': str(best_params)
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, 'hyperparameter_tuning.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
