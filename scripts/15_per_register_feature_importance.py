#!/usr/bin/env python3
"""
Per-register feature importance analysis using the 31-feature model.
Shows which features matter most for each register (academic, news, social, creative).
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from tool.feature_extractor import ALL_FEATURE_COLS

RANDOM_SEED = 42


def main():
    # Load 31-feature model
    model_path = os.path.join(MODELS_DIR, 'detector_all_31.joblib')
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run script 11 first.")
        return

    model = joblib.load(model_path)
    print(f"Loaded 31-feature model from {model_path}")

    # Load 31-feature corpus
    feat_path = os.path.join(DATA_DIR, 'corpus_features_31.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} texts from {feat_path}")

    # Drop rows with missing features
    df = df.dropna(subset=ALL_FEATURE_COLS)
    print(f"After dropping NA: {len(df)} texts")

    results = []

    for register in ['academic', 'news', 'social', 'creative']:
        print(f"\n=== Register: {register} ===")
        df_reg = df[df['register'] == register]
        if len(df_reg) < 1000:
            print(f"Skipping {register} (n={len(df_reg)} < 1000)")
            continue

        X = df_reg[ALL_FEATURE_COLS].values
        y = df_reg['label'].values

        # AUC
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        print(f"  n={len(df_reg)}, AUC={auc:.4f}")

        # Permutation importance (sample to speed up)
        n_eval = min(5000, len(df_reg))
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), n_eval, replace=False)
        perm_imp = permutation_importance(
            model, X[idx], y[idx], n_repeats=5,
            scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1
        )

        # Top 10 features
        imp_rows = []
        for i, col in enumerate(ALL_FEATURE_COLS):
            imp_rows.append({
                'register': register,
                'feature': col,
                'importance_mean': perm_imp.importances_mean[i],
                'importance_std': perm_imp.importances_std[i],
            })
        imp_df = pd.DataFrame(imp_rows).sort_values('importance_mean', ascending=False)

        print("Top 10 features by permutation importance:")
        print(imp_df.head(10).to_string(index=False))

        results.append(imp_df)

    # Combine all registers
    all_imp = pd.concat(results, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, 'per_register_feature_importance.csv')
    all_imp.to_csv(out_path, index=False)
    print(f"\nSaved per-register importance to {out_path}")

    # Summary: top features across registers
    summary = all_imp.groupby('feature')['importance_mean'].mean().sort_values(ascending=False).head(20)
    print("\nTop 20 features across all registers (mean importance):")
    print(summary.to_string())

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'top_features_summary.csv')
    summary.to_csv(summary_path, header=['importance_mean'])
    print(f"\nSaved top features summary to {summary_path}")


if __name__ == '__main__':
    main()
