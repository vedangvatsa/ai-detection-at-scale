#!/usr/bin/env python3
"""Optimize RoBERTa OpenAI detector + stylometric model combination per benchmark."""
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

ROBERTA_NAME = "roberta-base-openai-detector"

def load_roberta():
    print(f"Loading {ROBERTA_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_NAME)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_roberta_prob(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # Label 0 = fake (AI), label 1 = real (human). Use AI probability.
            return float(probs[0][0].cpu())
    except Exception as e:
        print(f"RoBERTa error: {e}")
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

def evaluate_benchmark(name, df, tokenizer, model, device, stylo_model, max_total=2000):
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

        roberta = get_roberta_prob(text, tokenizer, model, device)
        stylo = get_stylometric_prob(text, stylo_model)
        records.append({'roberta': roberta, 'stylometric': stylo, 'label': label})

    data = pd.DataFrame(records)
    print(f"Evaluated {len(data)} samples")

    X = data[['roberta', 'stylometric']].values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    rob_train_auc = roc_auc_score(y_train, X_train[:, 0])
    rob_test_auc = roc_auc_score(y_test, X_test[:, 0])
    sty_train_auc = roc_auc_score(y_train, X_train[:, 1])
    sty_test_auc = roc_auc_score(y_test, X_test[:, 1])
    combined_train_auc = roc_auc_score(y_train, train_probs)
    combined_test_auc = roc_auc_score(y_test, test_probs)

    print(f"RoBERTa: train AUC {rob_train_auc:.4f}, test AUC {rob_test_auc:.4f}")
    print(f"Stylometric: train AUC {sty_train_auc:.4f}, test AUC {sty_test_auc:.4f}")
    print(f"Combined LR: train AUC {combined_train_auc:.4f}, test AUC {combined_test_auc:.4f}")
    print(f"LR coefs: {clf.coef_.flatten()}")

    return {
        'benchmark': name,
        'roberta_test_auc': rob_test_auc,
        'stylometric_test_auc': sty_test_auc,
        'combined_test_auc': combined_test_auc,
        'lr_coef_roberta': float(clf.coef_[0][0]),
        'lr_coef_stylometric': float(clf.coef_[0][1]),
        'n_samples': len(data)
    }

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
    tokenizer, model, device = load_roberta()
    stylo_model = load_stylometric_model()

    results = []
    for df, name in [load_mage(), load_hc3(), load_turingbench()]:
        if df is None:
            continue
        results.append(evaluate_benchmark(name, df, tokenizer, model, device, stylo_model))

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'roberta_stylometric_optimized_results.csv'), index=False)
    print("\nSaved to results/roberta_stylometric_optimized_results.csv")

if __name__ == '__main__':
    main()
