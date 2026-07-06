#!/usr/bin/env python3
"""Analyze per-model performance on TuringBench."""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
import joblib

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features

FEATURE_COLS = ['mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                'connector_density', 'hedge_density', 'mean_sent_len',
                'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy']

def main():
    print("Loading model...")
    model = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))['model']

    print("Loading TuringBench...")
    tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    df = tb['train'].to_pandas()

    print(f"Total rows: {len(df)}")
    print("Label distribution:")
    print(df['label'].value_counts())

    # Sample each label evenly up to 500 per label for faster analysis
    max_per_label = 500
    sampled_frames = []
    for name, group in df.groupby('label'):
        n = min(max_per_label, len(group))
        sampled_frames.append(group.sample(n, random_state=42))
    sampled = pd.concat(sampled_frames).reset_index(drop=True)
    print(f"\nSampled {len(sampled)} rows")
    print(f"Sampled columns: {list(sampled.columns)}")

    records = []
    for i, row in sampled.iterrows():
        text = row['Generation']
        label_raw = row['label']
        label = 0 if label_raw == 'human' else 1
        model_name = label_raw

        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        feats = extract_features(text, extended=False)
        if feats is None:
            continue

        X = np.array([feats[k] for k in FEATURE_COLS])
        prob = model.predict_proba([X])[0][1]

        records.append({
            'model': model_name,
            'label': label,
            'prob': prob,
        })

    df_eval = pd.DataFrame(records)
    print(f"\nEvaluated {len(df_eval)} samples")

    # Overall AUC
    overall_auc = roc_auc_score(df_eval['label'], df_eval['prob'])
    print(f"Overall AUC: {overall_auc:.4f}")

    # Per-model AUC (human vs each AI model)
    per_model = []
    for ai_model in df_eval['model'].unique():
        if ai_model == 'human':
            continue
        subset = df_eval[(df_eval['model'] == ai_model) | (df_eval['model'] == 'human')]
        if len(subset) > 0:
            auc = roc_auc_score(subset['label'], subset['prob'])
            per_model.append({
                'ai_model': ai_model,
                'auc': auc,
                'n_samples': len(subset)
            })

    per_model_df = pd.DataFrame(per_model).sort_values('auc', ascending=False)
    print("\nPer-model AUC (human vs AI model):")
    print(per_model_df.to_string(index=False))

    per_model_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'turingbench_per_model.csv'), index=False)
    print("\nSaved to results/turingbench_per_model.csv")

if __name__ == '__main__':
    main()
