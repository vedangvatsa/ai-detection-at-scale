#!/usr/bin/env python3
"""Run TuringBench benchmark locally using the existing stylometric model."""
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

def load_turingbench():
    tb = None
    test_df = None
    dataset_name = 'TuringBench'
    try:
        print("Loading TuringBench...")
        tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    except Exception as e:
        print(f"Primary load failed: {e}")
        print("Trying alternative mirror...")
        try:
            tb = load_dataset("liam-hp/TuringBench")
        except Exception as e2:
            print(f"Mirror also failed: {e2}")
            print("Falling back to HC3...")
            hc3 = load_dataset("Hello-SimpleAI/HC3", name="default", revision="refs/convert/parquet")
            rows = []
            for item in hc3['train']:
                for ans in item.get('human_answers', []):
                    if isinstance(ans, str) and len(ans.strip()) > 20:
                        rows.append({'text': ans, 'label': 'human'})
                for ans in item.get('chatgpt_answers', []):
                    if isinstance(ans, str) and len(ans.strip()) > 20:
                        rows.append({'text': ans, 'label': 'chatgpt'})
            test_df = pd.DataFrame(rows)
            dataset_name = 'HC3'
            tb = None

    if tb is not None:
        print(f"Columns: {tb['train'].column_names}")
        print(f"Labels: {set(tb['train']['label'][:100])}")
        test_df = tb['validation'].to_pandas() if 'validation' in tb else tb['train'].to_pandas()

    return test_df, dataset_name

def main():
    print("Loading model...")
    model_data = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))
    model = model_data['model'] if isinstance(model_data, dict) else model_data

    test_df, dataset_name = load_turingbench()

    sample_size = min(5000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)
    print(f"Evaluating on {len(test_sample)} texts")

    features = []
    labels = []
    for i, row in test_sample.iterrows():
        text = row['text'] if 'text' in row else row.get('generation', row.get('Generation', ''))
        label_raw = row['label']
        if isinstance(label_raw, str):
            label = 0 if label_raw.lower() in ('human', '0') else 1
        else:
            label = 0 if int(label_raw) == 0 else 1

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
    results.to_csv(os.path.join(PROJECT_DIR, 'results', 'turingbench_results.csv'), index=False)
    print("Results saved")

if __name__ == '__main__':
    main()
