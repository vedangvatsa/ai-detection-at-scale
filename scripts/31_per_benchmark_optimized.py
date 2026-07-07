#!/usr/bin/env python3
"""Per-benchmark optimized detector selection with 2000 samples."""
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

MAX_LENGTH = 512
MAGE_MAX_LENGTH = 1024

def _get_device():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

def load_detector(model_name, ai_label=0):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = _get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device, ai_label

def detector_probs(texts, tokenizer, model, device, ai_label, max_length=MAX_LENGTH):
    probs = []
    for i, text in enumerate(texts):
        if i > 0 and i % 100 == 0:
            print(f"  inferred {i}/{len(texts)}...")
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                p = torch.softmax(logits, dim=-1)[0][ai_label].cpu().item()
                probs.append(float(p))
        except Exception as e:
            print(f"Detector error at {i}: {e}")
            probs.append(0.5)
    return probs

def load_stylometric_model():
    print("Loading stylometric model...")
    model_data = joblib.load(os.path.join(PROJECT_DIR, 'models', 'detector_all.joblib'))
    return model_data['model'] if isinstance(model_data, dict) else model_data

def stylometric_probs(texts, model):
    probs = []
    for i, text in enumerate(texts):
        if i > 0 and i % 100 == 0:
            print(f"  stylometric {i}/{len(texts)}...")
        try:
            feats = extract_features(text, extended=False)
            if feats is None:
                probs.append(0.5)
                continue
            X = np.array([feats[k] for k in FEATURE_COLS]).reshape(1, -1)
            probs.append(float(model.predict_proba(X)[0][1]))
        except Exception as e:
            print(f"Stylometric error at {i}: {e}")
            probs.append(0.5)
    return probs

def evaluate_mage(df, stylo_model, max_total=2000):
    print("\n=== MAGE (roberta-base + stylometric, 1024 tokens) ===")
    sample = df.sample(n=min(max_total, len(df)), random_state=42)
    texts = sample['text'].tolist()
    labels = sample['label'].values

    tokenizer, model, device, ai_label = load_detector("roberta-base-openai-detector", 0)
    base_probs = detector_probs(texts, tokenizer, model, device, ai_label, max_length=MAGE_MAX_LENGTH)
    del model, tokenizer
    torch_gc()

    stylo_probs = stylometric_probs(texts, stylo_model)
    X = np.column_stack([base_probs, stylo_probs])
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    test_probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    print(f"MAGE AUC: {auc:.4f} (n={len(y)})")
    print(f"LR coefs: {clf.coef_.flatten()}")
    return {'benchmark': 'MAGE', 'auc': auc, 'n_samples': len(y), 'detectors': ['roberta-base', 'stylometric'], 'coefs': list(clf.coef_.flatten())}

def evaluate_hc3(df, max_total=2000):
    print("\n=== HC3 (chatgpt-detector, 512 tokens) ===")
    sample = df.sample(n=min(max_total, len(df)), random_state=42)
    texts = sample['text'].tolist()
    labels = sample['label'].values

    tokenizer, model, device, ai_label = load_detector("Hello-SimpleAI/chatgpt-detector-roberta", 1)
    probs = detector_probs(texts, tokenizer, model, device, ai_label, max_length=MAX_LENGTH)
    del model, tokenizer
    torch_gc()

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, [1 if p >= 0.5 else 0 for p in probs])
    print(f"HC3 AUC: {auc:.4f}, Acc: {acc:.4f} (n={len(labels)})")
    return {'benchmark': 'HC3', 'auc': auc, 'accuracy': acc, 'n_samples': len(labels), 'detectors': ['chatgpt-detector']}

def evaluate_turingbench(df, max_total=2000):
    print("\n=== TuringBench (roberta-large, 512 tokens) ===")
    sample = df.sample(n=min(max_total, len(df)), random_state=42)
    texts = sample['text'].tolist()
    labels = sample['label'].values

    tokenizer, model, device, ai_label = load_detector("roberta-large-openai-detector", 0)
    probs = detector_probs(texts, tokenizer, model, device, ai_label, max_length=MAX_LENGTH)
    del model, tokenizer
    torch_gc()

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, [1 if p >= 0.5 else 0 for p in probs])
    print(f"TuringBench AUC: {auc:.4f}, Acc: {acc:.4f} (n={len(labels)})")
    return {'benchmark': 'TuringBench', 'auc': auc, 'accuracy': acc, 'n_samples': len(labels), 'detectors': ['roberta-large']}

def torch_gc():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_mage():
    try:
        mage = load_dataset("yaful/MAGE")
        df = mage['test'].to_pandas()
        df['label'] = 1 - df['label'].astype(int)
        return df
    except Exception as e:
        print(f"Could not load MAGE: {e}")
        return None

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
    return pd.DataFrame(rows)

def load_turingbench():
    try:
        tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
        df = tb['train'].to_pandas()
        df['label'] = df['label'].apply(lambda x: 0 if x == 'human' else 1)
        df['text'] = df['Generation']
        return df
    except Exception as e:
        print(f"Could not load TuringBench: {e}")
        return None

def main():
    stylo_model = load_stylometric_model()
    results = []

    mage_df = load_mage()
    if mage_df is not None:
        results.append(evaluate_mage(mage_df, stylo_model))

    hc3_df = load_hc3()
    results.append(evaluate_hc3(hc3_df))

    tb_df = load_turingbench()
    if tb_df is not None:
        results.append(evaluate_turingbench(tb_df))

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'per_benchmark_optimized.csv'), index=False)
    print("\nSaved to results/per_benchmark_optimized.csv")

if __name__ == '__main__':
    main()
