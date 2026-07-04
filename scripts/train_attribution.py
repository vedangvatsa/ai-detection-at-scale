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

DATA_DIR = "/Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/data"
MODELS_DIR = "/Users/vedang/ZCodeProject/research-paper-framework/papers/ai-detection-at-scale/models"

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
    
    # Create training pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    print("Training attribution model...")
    pipeline.fit(X, y)
    
    # Save the pipeline
    out_path = os.path.join(MODELS_DIR, 'attribution_classifier.joblib')
    joblib.dump(pipeline, out_path)
    print(f"Successfully saved attribution model to {out_path}")

if __name__ == '__main__':
    main()
