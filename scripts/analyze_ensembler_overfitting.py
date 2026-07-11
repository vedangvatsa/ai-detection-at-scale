#!/usr/bin/env python3
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS
from tool.neural_detector import compute_perplexity_and_burstiness, compute_binoculars_score
from tool.register_classifier import load_models_from_manifest, classify_register

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ONNX_PATH = os.path.join(MODELS_DIR, 'roberta_large_onnx_quantized.onnx')

def main():
    # 1. Load ONNX and tokenizer
    import onnxruntime as ort
    from transformers import AutoTokenizer
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    pytorch_path = os.path.join(MODELS_DIR, 'roberta_large_semantic_model')
    tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
    models_dict = load_models_from_manifest(MODELS_DIR)

    # 2. Load dataset
    beemo = load_dataset("toloka/beemo")
    df = beemo['train'].to_pandas()

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    test_df = df.loc[test_idx]

    def extract_pairs(dataframe):
        records = []
        for _, row in dataframe.iterrows():
            h = row['human_output']
            m = row['model_output']
            if isinstance(h, str) and len(h.strip()) >= 20:
                records.append({'text': h, 'label': 0})
            if isinstance(m, str) and len(m.strip()) >= 20:
                records.append({'text': m, 'label': 1})
        return pd.DataFrame(records)

    test_data = extract_pairs(test_df)
    val_size = min(3000, int(len(test_data) * 0.6))
    test_size = min(1500, len(test_data) - val_size)
    val_sample, test_sample_eval = train_test_split(test_data, train_size=val_size, test_size=test_size, random_state=42)

    def get_roberta_prob(text):
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding="max_length")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        logits = ort_session.run(None, ort_inputs)[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return float(probs[0][1])

    # Extract all features for validation and test samples (doing it once)
    print("Extracting validation features...")
    val_records = []
    for _, row in val_sample.iterrows():
        text = row['text']
        feats = extract_features(text, extended=False)
        if feats is None: continue
        stylo = [feats[k] for k in ORIGINAL_FEATURE_COLS]
        try:
            n = compute_perplexity_and_burstiness(text)
            ppl, burst = n['perplexity'], n['burstiness']
        except: ppl, burst = 50.0, 1.0
        try: bino = compute_binoculars_score(text)
        except: bino = 0.95
        prob = get_roberta_prob(text)
        val_records.append({'stylo': stylo, 'neural': [ppl, burst, bino], 'roberta': prob, 'label': row['label']})

    print("Extracting test features...")
    test_records = []
    for _, row in test_sample_eval.iterrows():
        text = row['text']
        feats = extract_features(text, extended=False)
        if feats is None: continue
        stylo = [feats[k] for k in ORIGINAL_FEATURE_COLS]
        try:
            n = compute_perplexity_and_burstiness(text)
            ppl, burst = n['perplexity'], n['burstiness']
        except: ppl, burst = 50.0, 1.0
        try: bino = compute_binoculars_score(text)
        except: bino = 0.95
        prob = get_roberta_prob(text)
        test_records.append({'stylo': stylo, 'neural': [ppl, burst, bino], 'roberta': prob, 'label': row['label']})

    # Prepare datasets for different feature combinations
    # 1. RoBERTa alone
    y_val = np.array([r['label'] for r in val_records])
    y_test = np.array([r['label'] for r in test_records])
    
    roberta_val = np.array([[r['roberta']] for r in val_records])
    roberta_test = np.array([[r['roberta']] for r in test_records])
    
    # RoBERTa alone AUC
    print(f"\nRoBERTa Alone Validation AUC: {roc_auc_score(y_val, roberta_val[:, 0]):.4f}")
    print(f"RoBERTa Alone Test AUC:       {roc_auc_score(y_test, roberta_test[:, 0]):.4f}")
    
    # 2. Full features (11 stylo + 3 neural + 1 roberta = 15 features)
    X_val_full = np.array([r['stylo'] + r['neural'] + [r['roberta']] for r in val_records])
    X_test_full = np.array([r['stylo'] + r['neural'] + [r['roberta']] for r in test_records])
    
    # 3. Only Neural + RoBERTa (ppl, burst, bino, roberta = 4 features)
    X_val_neural_rob = np.array([r['neural'] + [r['roberta']] for r in val_records])
    X_test_neural_rob = np.array([r['neural'] + [r['roberta']] for r in test_records])
    
    # Test different regularization strengths on Full features
    print("\n--- Tuning Full Features (15 feats) ---")
    for c in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]:
        clf = LogisticRegression(max_iter=1000, C=c)
        clf.fit(X_val_full, y_val)
        train_auc = roc_auc_score(y_val, clf.predict_proba(X_val_full)[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test_full)[:, 1])
        print(f"C={c:<6} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")
        
    # Test different regularization strengths on Neural + RoBERTa only
    print("\n--- Tuning Neural + RoBERTa (4 feats) ---")
    for c in [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]:
        clf = LogisticRegression(max_iter=1000, C=c)
        clf.fit(X_val_neural_rob, y_val)
        train_auc = roc_auc_score(y_val, clf.predict_proba(X_val_neural_rob)[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test_neural_rob)[:, 1])
        print(f"C={c:<6} | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}")

if __name__ == '__main__':
    main()
