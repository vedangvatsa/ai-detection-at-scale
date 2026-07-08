#!/usr/bin/env python3
"""
Trains a lightweight multiclass classifier for model attribution.
Predicts the source model group (human, openai, llama, mistral, cohere, mpt)
using the 11 stylometric features.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

def map_model_group(model_name):
    if not isinstance(model_name, str):
        return 'human'
    model_name = model_name.lower()
    if 'human' in model_name:
        return 'human'
    elif 'gpt4' in model_name or 'chatgpt' in model_name or 'gpt-3' in model_name or 'gpt3' in model_name or 'gpt2' in model_name:
        return 'openai'
    elif 'llama' in model_name:
        return 'llama'
    elif 'mistral' in model_name:
        return 'mistral'
    elif 'cohere' in model_name:
        return 'cohere'
    elif 'mpt' in model_name:
        return 'mpt'
    else:
        return 'other'

def main():
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} does not exist.")
        return

    print("Loading features parquet...")
    df = pd.read_parquet(feat_path, columns=['model'] + FEATURE_COLS)
    
    # Drop rows with NaN in features
    df = df.dropna(subset=FEATURE_COLS)
    
    print("Mapping models to groups...")
    df['group'] = df['model'].apply(map_model_group)
    
    # Sample from each group to build a balanced training set
    print("Sampling balanced dataset...")
    groups = df.groupby('group')
    sampled_dfs = []
    for g_name, g_df in groups:
        sample_size = min(30000, len(g_df))
        sampled_dfs.append(g_df.sample(sample_size, random_state=42))
    
    train_df = pd.concat(sampled_dfs)
    print(f"Total training samples: {len(train_df)}")
    print(train_df['group'].value_counts())
    
    X = train_df[FEATURE_COLS].values
    y = train_df['group'].values
    
    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Create training pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training attribution model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate on held-out test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHeld-out accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(train_df['group'].unique())))
    
    # Save the pipeline
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, 'attribution_classifier.joblib')
    joblib.dump(pipeline, out_path)
    print(f"\nSuccessfully saved attribution model to {out_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'classification_report': classification_report(y_test, y_pred, digits=4, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=sorted(train_df['group'].unique())).tolist(),
        'labels': sorted(train_df['group'].unique().tolist()),
    }
    metrics_path = os.path.join(MODELS_DIR, 'attribution_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved attribution metrics to {metrics_path}")

if __name__ == '__main__':
    main()
