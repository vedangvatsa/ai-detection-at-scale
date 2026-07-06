#!/usr/bin/env python3
"""Run MAGE/HC3/TuringBench with the available BERT semantic model + stylometric detector."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
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

def load_semantic_model():
    print("Loading semantic model (beemo_semantic_model)...")
    model_path = os.path.join(PROJECT_DIR, 'models', 'beemo_semantic_model')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_semantic_prob(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0][1].cpu())
    except Exception as e:
        print(f"Semantic model error: {e}")
        return 0.5

def load_stylometric_model():
    print("Loading stylometric model (detector_all)...")
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

def evaluate_benchmark(name, df, tokenizer, model, device, stylo_model, max_samples=5000):
    print(f"\n=== {name} ===")
    sample_size = min(max_samples, len(df))
    sample = df.sample(n=sample_size, random_state=42)

    semantic_probs = []
    stylo_probs = []
    labels = []

    for i, row in sample.iterrows():
        text = row.get('text', row.get('Generation', ''))
        label = row['label']

        if not isinstance(text, str) or len(text.strip()) < 20:
            continue

        semantic_probs.append(get_semantic_prob(text, tokenizer, model, device))
        stylo_probs.append(get_stylometric_prob(text, stylo_model))
        labels.append(label)

    labels = np.array(labels)
    semantic_probs = np.array(semantic_probs)
    stylo_probs = np.array(stylo_probs)
    hybrid_probs = 0.5 * semantic_probs + 0.5 * stylo_probs

    def report(probs, model_name):
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        print(f"{model_name} AUC: {auc:.4f}, Accuracy: {acc:.4f}")
        return auc, acc

    sem_auc, sem_acc = report(semantic_probs, 'Semantic (BERT)')
    sty_auc, sty_acc = report(stylo_probs, 'Stylometric')
    hyb_auc, hyb_acc = report(hybrid_probs, 'Hybrid (0.5 BERT + 0.5 stylometric)')

    return {
        'benchmark': name,
        'semantic_auc': sem_auc,
        'stylometric_auc': sty_auc,
        'hybrid_auc': hyb_auc,
        'n_samples': len(labels)
    }

def load_mage():
    try:
        mage = load_dataset("yaful/MAGE")
        df = mage['test'].to_pandas()
        # MAGE labels: 0=machine, 1=human. Flip to 0=human, 1=AI.
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
    tokenizer, model, device = load_semantic_model()
    stylo_model = load_stylometric_model()

    results = []
    for df, name in [load_mage(), load_hc3(), load_turingbench()]:
        if df is None:
            continue
        results.append(evaluate_benchmark(name, df, tokenizer, model, device, stylo_model))

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'hybrid_benchmark_results.csv'), index=False)
    print("\nSaved to results/hybrid_benchmark_results.csv")

if __name__ == '__main__':
    main()
