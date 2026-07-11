#!/usr/bin/env python3
import os
import sys
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import local modules
from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS
from tool.neural_detector import compute_perplexity_and_burstiness, compute_binoculars_score
from tool.register_classifier import load_models_from_manifest, classify_register

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ONNX_PATH = os.path.join(MODELS_DIR, 'roberta_large_onnx_quantized.onnx')

def main():
    # 1. Ensure datasets package is installed
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not found. Installing via pip...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    # 2. Load and verify ONNX session
    print("Initializing ONNX Inference Session...")
    import onnxruntime as ort
    from transformers import AutoTokenizer
    
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}. Please run export first.")
        
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    
    pytorch_path = os.path.join(MODELS_DIR, 'roberta_large_semantic_model')
    if os.path.exists(pytorch_path):
        tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        
    # Load register classifier models
    print("Loading register classifier...")
    models_dict = load_models_from_manifest(MODELS_DIR)

    # 3. Fetch Beemo Dataset
    print("Loading Toloka Beemo dataset...")
    beemo = load_dataset("toloka/beemo")
    df = beemo['train'].to_pandas()

    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    test_df = df.loc[train_idx] # Wait, let's make sure we split test_df correctly. Wait, train_idx, test_idx = split(df.index). test_df is df.loc[test_idx].
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
    print(f"Total available test pairs: {len(test_data)}")

    # We match the kaggle notebook's splits
    val_size = min(3000, int(len(test_data) * 0.6))
    test_size = min(1500, len(test_data) - val_size)
    val_sample, test_sample_eval = train_test_split(test_data, train_size=val_size, test_size=test_size, random_state=42)

    print(f"Validation set size: {len(val_sample)}, Test set size: {len(test_sample_eval)}")

    def get_roberta_prob(text):
        try:
            inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=256, padding="max_length")
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            logits = ort_session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return float(probs[0][1])
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return 0.5

    # 4. Extract features on Validation Set
    records = []
    print("Extracting features on validation set (this will take ~10-15 minutes on CPU)...")
    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample)):
        text = row['text']
        label = row['label']
        
        feats = extract_features(text, extended=False)
        if feats is None:
            continue
        
        stylo_vector = [feats[k] for k in ORIGINAL_FEATURE_COLS]
        
        try:
            neural = compute_perplexity_and_burstiness(text)
            ppl = neural['perplexity']
            burst = neural['burstiness']
        except:
            ppl, burst = 50.0, 1.0
            
        try:
            bino = compute_binoculars_score(text)
        except:
            bino = 0.95
            
        roberta_prob = get_roberta_prob(text)
        reg, _ = classify_register(stylo_vector, models_dict)
        
        records.append({
            'vector': stylo_vector + [ppl, burst, bino, roberta_prob],
            'label': label,
            'register': reg
        })

    # 5. Fit Logistic Regression Ensemblers
    print("\nTraining ensemblers...")
    ensemblers = {}
    X_all = np.array([r['vector'] for r in records])
    y_all = np.array([r['label'] for r in records])

    ensembler_all = LogisticRegression(max_iter=1000)
    ensembler_all.fit(X_all, y_all)
    ensemblers['all'] = ensembler_all
    print(f"Overall Train AUC: {roc_auc_score(y_all, ensembler_all.predict_proba(X_all)[:, 1]):.4f}")

    df_val = pd.DataFrame(records)
    for reg in ['news', 'academic', 'social', 'creative']:
        reg_records = df_val[df_val['register'] == reg]
        if len(reg_records) >= 50:
            X_reg = np.array(reg_records['vector'].tolist())
            y_reg = np.array(reg_records['label'].tolist())
            
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_reg, y_reg)
            ensemblers[reg] = clf
            print(f"Register '{reg}' Train AUC: {roc_auc_score(y_reg, clf.predict_proba(X_reg)[:, 1]):.4f}")
        else:
            ensemblers[reg] = ensembler_all
            print(f"Register '{reg}' fallback to overall model (not enough examples: {len(reg_records)})")

    # Backup old ensembler
    backup_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib.bak')
    dest_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')
    if os.path.exists(dest_path):
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(dest_path, backup_path)
        print(f"Backed up old ensembler to {backup_path}")

    # Save new ensembler
    joblib.dump(ensemblers, dest_path)
    print(f"Saved new ensembler to {dest_path}")

    # 6. Evaluate on Test Set
    X_test_all = []
    y_test_all = []
    y_test_preds = []

    print("\nEvaluating Register-Aware Ensembler on held-out test cases...")
    for _, row in tqdm(test_sample_eval.iterrows(), total=len(test_sample_eval)):
        text = row['text']
        label = row['label']
        
        feats = extract_features(text, extended=False)
        if feats is None:
            continue
        
        stylo_vector = [feats[k] for k in ORIGINAL_FEATURE_COLS]
        
        try:
            neural = compute_perplexity_and_burstiness(text)
            ppl = neural['perplexity']
            burst = neural['burstiness']
        except:
            ppl, burst = 50.0, 1.0
            
        try:
            bino = compute_binoculars_score(text)
        except:
            bino = 0.95
            
        roberta_prob = get_roberta_prob(text)
        reg, _ = classify_register(stylo_vector, models_dict)
        
        vec = stylo_vector + [ppl, burst, bino, roberta_prob]
        clf = ensemblers.get(reg, ensemblers['all'])
        pred_prob = clf.predict_proba([vec])[0][1]
        
        X_test_all.append(vec)
        y_test_all.append(label)
        y_test_preds.append(pred_prob)

    test_auc = roc_auc_score(y_test_all, y_test_preds)
    print(f"\n--- FINAL SOTA PIPELINE TEST AUC: {test_auc:.4f} ---")

if __name__ == '__main__':
    main()
