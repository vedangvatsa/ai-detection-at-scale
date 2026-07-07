#!/usr/bin/env python3
"""Advanced ensemble: roberta-base, roberta-large, chatgpt-detector + stylometric, with 512 tokens."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features

FEATURE_COLS = ['mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
                'connector_density', 'hedge_density', 'mean_sent_len',
                'boost_density', 'char_entropy', 'rep_rate', 'punct_entropy']

MODEL_CONFIGS = {
    "roberta-base-openai": ("roberta-base-openai-detector", 0),
    "roberta-large-openai": ("roberta-large-openai-detector", 0),
    "chatgpt-detector": ("Hello-SimpleAI/chatgpt-detector-roberta", 1),
}

MAX_LENGTH = 512

def _get_device():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def load_detector(name):
    print(f"Loading {name}...")
    model_name, ai_label = MODEL_CONFIGS[name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = _get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device, ai_label

def get_detector_prob(text, tokenizer, model, device, ai_label):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0][ai_label].cpu())
    except Exception as e:
        print(f"Detector error: {e}")
        return 0.5

def load_stylometric_model():
    print("Loading stylometric model...")
    model_data = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))
    return model_data['model'] if isinstance(model_data, dict) else model_data

def get_stylometric_prob(text, model):
    try:
        feats = extract_features(text, extended=False)
        if feats is None:
            return 0.5
        X = np.array([feats[k] for k in FEATURE_COLS]).reshape(1, -1)
        return model.predict_proba(X)[0][1]
    except Exception as e:
        print(f"Stylometric error: {e}")
        return 0.5

def evaluate_benchmark(name, df, detectors, stylo_model, max_total=1000):
    print(f"\n=== {name} ===")
    sample_size = min(max_total, len(df))
    sample = df.sample(n=sample_size, random_state=42)

    records = []
    for idx, (i, row) in enumerate(sample.iterrows()):
        if idx > 0 and idx % 100 == 0:
            print(f"  Processed {idx}/{len(sample)}...")
        text = row.get('text', row.get('Generation', ''))
        label = row['label']

        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        rec = {'label': label}
        for det_name, (tokenizer, model, device, ai_label) in detectors.items():
            rec[det_name] = get_detector_prob(text, tokenizer, model, device, ai_label)
        rec['stylometric'] = get_stylometric_prob(text, stylo_model)
        records.append(rec)

    data = pd.DataFrame(records)
    print(f"Evaluated {len(data)} samples")

    feature_cols = list(detectors.keys()) + ['stylometric']
    X = data[feature_cols].values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    result = {'benchmark': name, 'combined_train_auc': roc_auc_score(y_train, train_probs), 'combined_test_auc': roc_auc_score(y_test, test_probs), 'n_samples': len(data)}
    for col in feature_cols:
        result[f'{col}_test_auc'] = roc_auc_score(y_test, X_test[:, feature_cols.index(col)])
    for idx, col in enumerate(feature_cols):
        result[f'lr_coef_{col}'] = float(clf.coef_[0][idx])

    print(f"Combined LR: train AUC {result['combined_train_auc']:.4f}, test AUC {result['combined_test_auc']:.4f}")
    print(f"LR coefs: {clf.coef_.flatten()}")
    for col in feature_cols:
        print(f"{col} test AUC: {result[f'{col}_test_auc']:.4f}")

    return result

def load_mage():
    try:
        mage = load_dataset("yaful/MAGE")
        df = mage['test'].to_pandas()
        df['label'] = 1 - df['label'].astype(int)
        return df, 'MAGE'
    except Exception as e:
        print(f"Could not load MAGE: {e}")
        return None, None

def load_hc3():
    hc3 = load_dataset("Hello-SimpleAI/HC3", name="default", revision="refs/convert/parquet")
    rows = []
    for item in hc3['train']:
        for ans in item.get('human_answers', []):
            if isinstance(ans, str) and len(ans.strip()) > 20:
                rows.append({'text': ans, 'label': 0})
        for ans in item.get('chatgpt_answers', []):
            if isinstance(ans, str) and len(ans.strip()) > 20:
                rows.append({'text': ans, 'label': 1})
    return pd.DataFrame(rows), 'HC3'

def load_turingbench():
    try:
        tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
        df = tb['train'].to_pandas()
        df['label'] = df['label'].apply(lambda x: 0 if x == 'human' else 1)
        df['text'] = df['Generation']
        return df, 'TuringBench'
    except Exception as e:
        print(f"Could not load TuringBench: {e}")
        return None, None

def main():
    detectors = {name: load_detector(name) for name in MODEL_CONFIGS}
    stylo_model = load_stylometric_model()

    results = []
    for df, name in [load_mage(), load_hc3(), load_turingbench()]:
        if df is None:
            continue
        results.append(evaluate_benchmark(name, df, detectors, stylo_model))
        # Free MPS cache between benchmarks
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'advanced_ensemble_results.csv'), index=False)
    print("\nSaved to results/advanced_ensemble_results.csv")

if __name__ == '__main__':
    main()
