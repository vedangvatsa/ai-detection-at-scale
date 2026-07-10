#!/usr/bin/env python3
"""
Train a specialized classifier to detect Reasoning Models (o1/o3/DeepSeek-R1).

Uses advanced syntactic features (noun_verb_ratio, adj_adv_ratio,
pos_transition_entropy, sent_length_std) to train a Random Forest classifier
specifically targeting reasoning model signatures.

Usage:
    python scripts/train_reasoning_detector.py
"""
import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

FEATURE_COLS = ['noun_verb_ratio', 'adj_adv_ratio', 'pos_transition_entropy', 'sent_length_std']

def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic feature data based on verified profiles:
    - Reasoning: Low POS entropy (5.23), high adj/adv (2.50), high noun/verb (1.00), low sent length variation (2.19)
    - Standard LLM: Medium POS entropy (5.49), medium adj/adv (1.33), medium noun/verb (0.76), medium sent length variation (3.15)
    - Human: High POS entropy (5.60), low adj/adv (0.71), low noun/verb (0.64), high sent length variation (3.25)
    """
    np.random.seed(seed)
    
    # 1. Reasoning models
    reasoning_features = np.column_stack([
        np.random.normal(1.00, 0.15, n_samples),  # noun_verb_ratio
        np.random.normal(2.50, 0.35, n_samples),  # adj_adv_ratio
        np.random.normal(5.23, 0.08, n_samples),  # pos_transition_entropy
        np.random.normal(2.19, 0.30, n_samples),  # sent_length_std
    ])
    
    # 2. Standard LLM
    std_llm_features = np.column_stack([
        np.random.normal(0.76, 0.12, n_samples),  # noun_verb_ratio
        np.random.normal(1.33, 0.25, n_samples),  # adj_adv_ratio
        np.random.normal(5.49, 0.06, n_samples),  # pos_transition_entropy
        np.random.normal(3.15, 0.35, n_samples),  # sent_length_std
    ])
    
    # 3. Human
    human_features = np.column_stack([
        np.random.normal(0.64, 0.10, n_samples),  # noun_verb_ratio
        np.random.normal(0.71, 0.18, n_samples),  # adj_adv_ratio
        np.random.normal(5.60, 0.05, n_samples),  # pos_transition_entropy
        np.random.normal(3.25, 0.40, n_samples),  # sent_length_std
    ])
    
    # Target: 1 = Reasoning model, 0 = Non-reasoning (Standard LLM or Human)
    X = np.vstack([reasoning_features, std_llm_features, human_features])
    y = np.concatenate([
        np.ones(n_samples),      # Reasoning
        np.zeros(n_samples),     # Standard LLM
        np.zeros(n_samples)      # Human
    ])
    
    return X, y

def main():
    print("Generating training dataset for Reasoning Model detector...")
    X_train, y_train = generate_synthetic_data(n_samples=2000, seed=42)
    X_test, y_test = generate_synthetic_data(n_samples=500, seed=100)
    
    print(f"Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, probs)
    print(f"\nModel Performance on Test Set:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["Human/Standard LLM", "Reasoning LLM"]))
    
    # Feature Importance
    print("Feature Importances:")
    for name, imp in zip(FEATURE_COLS, clf.feature_importances_):
        print(f"  {name:25s}: {imp:.4f}")
        
    save_path = os.path.join(MODELS_DIR, "detector_reasoning.joblib")
    joblib.dump({
        'model': clf,
        'feature_cols': FEATURE_COLS,
        'description': 'Reasoning model signature detector'
    }, save_path)
    print(f"\nReasoning detector saved to: {save_path}")

if __name__ == '__main__':
    main()
