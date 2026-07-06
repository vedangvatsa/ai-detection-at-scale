#!/usr/bin/env python3
"""Run MAGE/HC3 benchmark locally using the existing stylometric model."""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from datasets import load_dataset
import joblib

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features

FEATURE_COLS = ['mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                'connector_density', 'hedge_density', 'mean_sent_len',
                'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy']

def load_mage_or_hc3():
    try:
        print("Loading MAGE dataset...")
        mage = load_dataset("yaful/MAGE")
        print(f"Columns: {mage['test'].column_names}")
        test_df = mage['test'].to_pandas()
        return test_df, 'MAGE'
    except Exception as e:
        print(f"Could not load yaful/MAGE: {e}")
        print("Trying HC3...")
        mage = load_dataset("Hello-SimpleAI/HC3")
        print(f"Columns: {mage['train'].column_names}")
        rows = []
        for item in mage['train']:
            for ans in item.get('human_answers', []):
                if isinstance(ans, str) and len(ans.strip()) > 20:
                    rows.append({'text': ans, 'label': 0})
            for ans in item.get('chatgpt_answers', []):
                if isinstance(ans, str) and len(ans.strip()) > 20:
                    rows.append({'text': ans, 'label': 1})
        test_df = pd.DataFrame(rows)
        return test_df, 'HC3'

def main():
    print("Loading model...")
    model_data = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))
    model = model_data['model'] if isinstance(model_data, dict) else model_data

    test_df, dataset_name = load_mage_or_hc3()

    sample_size = min(5000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)
    print(f"Evaluating on {len(test_sample)} texts")

    features = []
    labels = []
    for i, row in test_sample.iterrows():
        text = row.get('text', row.get('article', ''))
        label_raw = row['label']
        if isinstance(label_raw, str):
            label = 0 if label_raw.lower() in ('human', '1') else 1
        else:
            label = 1 - int(label_raw)

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

    print(f"{dataset_name} AUC: {auc:.4f}")
    print(f"{dataset_name} Accuracy: {acc:.4f}")
    print(f"Samples evaluated: {len(labels)}")

    results = pd.DataFrame([{
        'benchmark': dataset_name,
        'auc': auc,
        'accuracy': acc,
        'n_samples': len(labels)
    }])
    results.to_csv(os.path.join(PROJECT_DIR, 'results', 'mage_hc3_results.csv'), index=False)
    print("Results saved")

if __name__ == '__main__':
    main()
