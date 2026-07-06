#!/usr/bin/env python3
"""Evaluate public RoBERTa OpenAI detector on MAGE/HC3/TuringBench."""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

MODEL_NAME = "roberta-base-openai-detector"

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_prob(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            # OpenAI detector: label 0 = fake, label 1 = real.
            # We want 1 = AI. Use fake probability.
            return float(probs[0][0].cpu())
    except Exception as e:
        print(f"Model error: {e}")
        return 0.5

def evaluate(name, df, tokenizer, model, device, max_samples=1000):
    print(f"\n=== {name} ===")
    sample = df.sample(n=min(max_samples, len(df)), random_state=42)
    probs = []
    labels = []
    for i, row in sample.iterrows():
        text = row.get('text', row.get('Generation', ''))
        label = row['label']
        if not isinstance(text, str) or len(text.strip()) < 20:
            continue
        probs.append(get_prob(text, tokenizer, model, device))
        labels.append(label)

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, Samples: {len(labels)}")
    return {'benchmark': name, 'auc': auc, 'accuracy': acc, 'n_samples': len(labels)}

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
    tokenizer, model, device = load_model()
    results = []
    for df, name in [load_mage(), load_hc3(), load_turingbench()]:
        if df is None:
            continue
        results.append(evaluate(name, df, tokenizer, model, device))

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(PROJECT_DIR, 'results', 'roberta_openai_detector_results.csv'), index=False)
    print("\nSaved to results/roberta_openai_detector_results.csv")

if __name__ == '__main__':
    main()
