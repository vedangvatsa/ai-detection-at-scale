#!/usr/bin/env python3
"""
Adversarial Training for Stylometric AI Text Detectors.

Loads the pre-extracted features from the corpus, generates perturbed adversarial
examples simulating humanizer/paraphrase attacks, and retrains the Random Forest
classifiers. This directly improves robustness against paraphrase attacks.

Usage:
    python scripts/adversarial_training.py
"""
import os
import sys
import shutil
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'corpus_features.parquet')

FEATURE_COLS = [
    "mtld", "sent_cv", "self_mention_density", "opener_ratio",
    "connector_density", "hedge_density", "mean_sent_len",
    "boost_density", "char_entropy", "rep_rate", "punct_entropy"
]

REGISTERS = ["academic", "news", "social", "creative"]
RANDOM_SEED = 42

def perturb_adversarial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate perturbed copies of AI-generated examples to simulate humanizers.
    - Reduce connector density (humanizers strip formal transitions).
    - Increase self-mention density (humanizers add first-person pronouns).
    - Increase vocabulary diversity / MTLD (synonym swapping increases unique words).
    """
    ai_df = df[df['label'] == 1].copy()
    if len(ai_df) == 0:
        return pd.DataFrame(columns=df.columns)
        
    np.random.seed(RANDOM_SEED)
    n = len(ai_df)
    
    # Apply perturbations
    ai_df['connector_density'] *= np.random.uniform(0.5, 0.8, n)
    ai_df['self_mention_density'] += np.random.uniform(0.005, 0.02, n)
    ai_df['mtld'] *= np.random.uniform(1.1, 1.3, n)
    ai_df['sent_cv'] *= np.random.uniform(0.8, 1.2, n)
    ai_df['hedge_density'] *= np.random.uniform(0.7, 1.0, n)
    ai_df['boost_density'] *= np.random.uniform(0.7, 1.0, n)
    
    # Mark as adversarial/humanized source
    ai_df['model'] = 'humanized_adversarial'
    
    return ai_df

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Parquet file not found at {DATA_PATH}")
        sys.exit(1)
        
    print(f"Loading corpus features from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    print("Generating adversarial perturbations for training set...")
    adv_df = perturb_adversarial(df)
    print(f"Generated {len(adv_df)} adversarial AI examples.")
    
    # Combine original and adversarial
    df_robust = pd.concat([df, adv_df], ignore_index=True)
    print(f"Total robust dataset size: {len(df_robust)} rows")
    
    # Backup existing models
    print("\nBacking up existing models...")
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.joblib') and not filename.endswith('.bak'):
            src = os.path.join(MODELS_DIR, filename)
            dst = os.path.join(MODELS_DIR, filename + '.bak')
            shutil.copyfile(src, dst)
            
    # Retrain register-specific classifiers
    detectors = {}
    print("\nTraining robust register-specific detectors...")
    for reg in REGISTERS:
        reg_df = df_robust[df_robust['register'] == reg]
        if len(reg_df) < 100:
            print(f"  Skipping {reg} due to insufficient data.")
            continue
            
        X = reg_df[FEATURE_COLS].values
        y = reg_df['label'].values
        
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        clf.fit(X, y)
        
        # Evaluate on original clean data
        clean_df = df[df['register'] == reg]
        X_clean = clean_df[FEATURE_COLS].values
        y_clean = clean_df['label'].values
        proba = clf.predict_proba(X_clean)[:, 1]
        auc = roc_auc_score(y_clean, proba)
        
        print(f"  Register: {reg:10s} | Size: {len(reg_df):6d} | Clean AUC: {auc:.4f}")
        
        model_path = os.path.join(MODELS_DIR, f"detector_{reg}.joblib")
        joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS, 'register': reg}, model_path)
        detectors[reg] = clf
        
    # Retrain all-register detector
    print("\nTraining robust all-register detector...")
    X_all = df_robust[FEATURE_COLS].values
    y_all = df_robust['label'].values
    
    all_clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    all_clf.fit(X_all, y_all)
    
    # Evaluate clean all
    proba_all = all_clf.predict_proba(df[FEATURE_COLS].values)[:, 1]
    auc_all = roc_auc_score(df['label'].values, proba_all)
    print(f"  All-Register | Size: {len(df_robust):6d} | Clean AUC: {auc_all:.4f}")
    
    all_model_path = os.path.join(MODELS_DIR, "detector_all.joblib")
    joblib.dump({'model': all_clf, 'feature_cols': FEATURE_COLS, 'register': 'all'}, all_model_path)
    
    print("\nAdversarial training completed and models saved successfully!")

if __name__ == '__main__':
    main()
