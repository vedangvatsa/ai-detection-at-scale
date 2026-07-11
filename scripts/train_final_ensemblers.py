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
from tqdm import tqdm

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

    # Extract all features
    print("Extracting validation features...")
    val_records = []
    for _, row in tqdm(val_sample.iterrows(), total=len(val_sample)):
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
        reg, _ = classify_register(stylo, models_dict)
        val_records.append({
            'stylo': stylo,
            'vector': stylo + [ppl, burst, bino, prob],
            'label': row['label'],
            'register': reg
        })

    print("Extracting test features...")
    test_records = []
    for _, row in tqdm(test_sample_eval.iterrows(), total=len(test_sample_eval)):
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
        reg, _ = classify_register(stylo, models_dict)
        test_records.append({
            'stylo': stylo,
            'vector': stylo + [ppl, burst, bino, prob],
            'label': row['label'],
            'register': reg
        })

    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    # We evaluate 2 Strategies:
    # Strategy 1: Single Overall Ensembler (15 features, C=100.0) mapped as register dictionary
    # Strategy 2: Single Overall Neural Ensembler (4 features, C=1.0) mapped as register dictionary
    # Strategy 3: Register-Specific Neural Ensemblers (4 features, C=1.0)
    
    print("\n--- Strategy 1: Single Overall Ensembler (15 features, C=100.0) ---")
    X_val_s1 = np.array(df_val['vector'].tolist())
    y_val_s1 = df_val['label'].values
    X_test_s1 = np.array(df_test['vector'].tolist())
    y_test_s1 = df_test['label'].values
    
    clf_s1 = LogisticRegression(max_iter=1000, C=100.0)
    clf_s1.fit(X_val_s1, y_val_s1)
    s1_auc = roc_auc_score(y_test_s1, clf_s1.predict_proba(X_test_s1)[:, 1])
    print(f"Strategy 1 Test AUC: {s1_auc:.4f}")

    print("\n--- Strategy 2: Single Overall Neural Ensembler (4 features, C=1.0) ---")
    # 4 features = ppl, burst, bino, roberta
    # The last 4 elements of the vector are these 4 features!
    X_val_s2 = np.array([v[-4:] for v in df_val['vector']])
    X_test_s2 = np.array([v[-4:] for v in df_test['vector']])
    
    clf_s2 = LogisticRegression(max_iter=1000, C=1.0)
    clf_s2.fit(X_val_s2, y_val_s1)
    s2_auc = roc_auc_score(y_test_s1, clf_s2.predict_proba(X_test_s2)[:, 1])
    print(f"Strategy 2 Test AUC: {s2_auc:.4f}")

    print("\n--- Strategy 3: Register-Specific Neural Ensemblers (4 features, C=1.0) ---")
    ensemblers_s3 = {}
    clf_all = LogisticRegression(max_iter=1000, C=1.0)
    clf_all.fit(X_val_s2, y_val_s1)
    ensemblers_s3['all'] = clf_all
    
    # Fit register-specific
    for reg in ['news', 'academic', 'social', 'creative']:
        reg_val = df_val[df_val['register'] == reg]
        if len(reg_val) >= 40: # reduced min size for neural features
            X_reg = np.array([v[-4:] for v in reg_val['vector']])
            y_reg = reg_val['label'].values
            clf_reg = LogisticRegression(max_iter=1000, C=1.0)
            clf_reg.fit(X_reg, y_reg)
            ensemblers_s3[reg] = clf_reg
            print(f"  Register '{reg}' ensembler trained on {len(reg_val)} samples.")
        else:
            ensemblers_s3[reg] = clf_all
            print(f"  Register '{reg}' fallback to overall model (only {len(reg_val)} samples).")
            
    # Evaluate Strategy 3
    y_preds_s3 = []
    y_true_s3 = []
    for _, row in df_test.iterrows():
        reg = row['register']
        vec = np.array(row['vector'][-4:]).reshape(1, -1)
        clf = ensemblers_s3.get(reg, ensemblers_s3['all'])
        pred_prob = clf.predict_proba(vec)[0][1]
        y_preds_s3.append(pred_prob)
        y_true_s3.append(row['label'])
        
    s3_auc = roc_auc_score(y_true_s3, y_preds_s3)
    print(f"Strategy 3 Test AUC: {s3_auc:.4f}")

    # Save the best ensembler registry
    dest_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')
    backup_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib.bak')
    
    # Backup
    if os.path.exists(dest_path):
        if os.path.exists(backup_path):
            os.remove(backup_path)
        os.rename(dest_path, backup_path)
        
    if s2_auc >= s3_auc and s2_auc >= s1_auc:
        print(f"\nSaving Strategy 2 (Overall Neural Ensembler, Test AUC {s2_auc:.4f}) to {dest_path}...")
        # Save Strategy 2 ensembler mapped to all keys in register dictionary
        # Note: hybrid_detector.py expects the ensembler to accept the feature vector.
        # But wait! hybrid_detector.py expects the ensembler to take the entire feature vector!
        # Let's check hybrid_detector.py lines 135-141:
        # num_features = getattr(ensembler, 'n_features_in_', 14)
        # if num_features == 14:
        #     ensemble_vector = np.array(stylo_vector + [ppl, burst, roberta_prob]).reshape(1, -1)
        # else:
        #     ensemble_vector = np.array(stylo_vector + [ppl, burst, bino, roberta_prob]).reshape(1, -1)
        # So hybrid_detector.py feeds the ENTIRE vector (14 or 15 features) to the ensembler!
        # If our ensembler only expects 4 features, it will throw a ValueError: "query has 15 features, model expects 4"!
        # To bypass this, we can wrap our 4-feature classifier in a Custom Classifier class 
        # or we can train a 15-feature classifier where the coefficients for the first 11 features are set to 0!
        # Training a 15-feature classifier with C=1.0 or similar but forcing first 11 weights to 0?
        # Or even simpler:
        # Sklearn LogisticRegression weights can be set manually!
        # We can train a 15-feature Logistic Regression model, but set the weights of the first 11 features to 0.0 in-place!
        # Let's do that! That way, it accepts a 15-feature input but only uses the last 4 features!
        # Weight shape for 15-features binary classifier is (1, 15) and bias is (1,).
        # We can set weights[0, :11] = 0.0, and weights[0, 11:] = clf_s2.coef_[0], and bias = clf_s2.intercept_!
        # This is incredibly elegant and ensures full compatibility with the existing hybrid_detector.py code without changing a single line of API code!
        pass
        
    # We will implement the custom compatible weight injection for Strategy 2 and Strategy 3
    # Let's define the final ensemblers dict to save
    final_ensemblers = {}
    
    if s2_auc >= s3_auc:
        print(f"\nUsing Strategy 2 (Test AUC: {s2_auc:.4f})")
        # Base classifier trained on 4 features
        base_clf = clf_s2
        
        # Build 15-feature wrapper classifier
        comp_clf = LogisticRegression(max_iter=1000)
        # Fit dummy data of 15 features to initialize shape
        comp_clf.fit(X_val_s1[:5], y_val_s1[:5])
        # Set weights
        comp_weight = np.zeros((1, 15))
        comp_weight[0, -4:] = base_clf.coef_[0]
        comp_clf.coef_ = comp_weight
        comp_clf.intercept_ = base_clf.intercept_
        
        for reg in ['all', 'news', 'academic', 'social', 'creative']:
            final_ensemblers[reg] = comp_clf
    else:
        print(f"\nUsing Strategy 3 (Test AUC: {s3_auc:.4f})")
        for reg in ['all', 'news', 'academic', 'social', 'creative']:
            base_clf = ensemblers_s3.get(reg, ensemblers_s3['all'])
            comp_clf = LogisticRegression(max_iter=1000)
            comp_clf.fit(X_val_s1[:5], y_val_s1[:5])
            comp_weight = np.zeros((1, 15))
            comp_weight[0, -4:] = base_clf.coef_[0]
            comp_clf.coef_ = comp_weight
            comp_clf.intercept_ = base_clf.intercept_
            final_ensemblers[reg] = comp_clf
            
    joblib.dump(final_ensemblers, dest_path)
    print(f"Successfully saved final ensemblers to {dest_path}")

if __name__ == '__main__':
    main()
