#!/usr/bin/env python3
"""Run HC3 benchmark locally using the existing stylometric model."""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from datasets import load_dataset
import joblib

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS

FEATURE_COLS = ORIGINAL_FEATURE_COLS

def main():
    print("Loading model...")
    model_data = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))
    model = model_data['model'] if isinstance(model_data, dict) else model_data

    print("Loading HC3 dataset...")
    hc3 = load_dataset("Hello-SimpleAI/HC3", name="default", revision="refs/convert/parquet")
    print(f"Columns: {hc3['train'].column_names}")

    rows = []
    for item in hc3['train']:
        for ans in item.get('human_answers', []):
            if isinstance(ans, str) and len(ans.strip()) > 20:
                rows.append({'text': ans, 'label': 0})
        for ans in item.get('chatgpt_answers', []):
            if isinstance(ans, str) and len(ans.strip()) > 20:
                rows.append({'text': ans, 'label': 1})

    test_df = pd.DataFrame(rows)

    sample_size = min(5000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)
    print(f"Evaluating on {len(test_sample)} texts")

    features = []
    labels = []
    for i, row in test_sample.iterrows():
        text = row['text']
        label = row['label']

        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        feats = extract_features(text, extended=False)
        if feats is None:
            continue

        X = np.array([feats[k] for k in FEATURE_COLS])
        features.append(X)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    probs = model.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)

    print(f"HC3 AUC: {auc:.4f}")
    print(f"HC3 Accuracy: {acc:.4f}")
    print(f"Samples evaluated: {len(labels)}")

    results = pd.DataFrame([{
        'benchmark': 'HC3',
        'auc': auc,
        'accuracy': acc,
        'n_samples': len(labels)
    }])
    results.to_csv(os.path.join(PROJECT_DIR, 'results', 'hc3_results.csv'), index=False)
    print("Results saved")

if __name__ == '__main__':
    main()
