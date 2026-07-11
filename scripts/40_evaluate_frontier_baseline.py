#!/usr/bin/env python3
"""
Phase 1A: Download Defactify dataset (frontier LLMs) and evaluate our
existing stylometric Random Forest detectors to measure the generalization gap.

Outputs:
    results/frontier_baseline.csv   — per-model & per-register AUC scores
    stdout                          — summary table
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS


MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')


def load_defactify():
    """Download and load the Defactify Text Dataset from Hugging Face."""
    from datasets import load_dataset

    print("Downloading Defactify_Text_Dataset from Hugging Face...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset")

    # Explore available splits and columns
    print(f"Available splits: {list(ds.keys())}")
    split_name = 'train' if 'train' in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"Sample row:\n{df.iloc[0]}")

    return df


def prepare_binary_labels(df):
    """
    Convert the Defactify dataset into binary human/AI labels.
    Uses 'Text' for text, 'Label_A' for binary labels (0=human, 1=ai),
    and 'Label_B' for the generator model name.
    """
    df['binary_label'] = df['Label_A'].astype(int)
    df['text_clean'] = df['Text'].astype(str)
    df['source_model'] = df['Label_B'].astype(str)
    df['is_human'] = (df['binary_label'] == 0)

    print(f"Using text column: 'Text', label column: 'Label_A', model column: 'Label_B'")
    print(f"Binary split: human={df['is_human'].sum()}, AI={(~df['is_human']).sum()}")

    return df


def extract_features_batch(texts, max_samples=5000):
    """Extract stylometric features for a list of texts."""
    features_list = []
    skipped = 0

    total = min(len(texts), max_samples)
    print(f"Extracting features for {total} texts...")

    for i, text in enumerate(texts[:max_samples]):
        if i % 500 == 0 and i > 0:
            print(f"  [{i}/{total}] extracted ({skipped} skipped)")

        feats = extract_features(text, extended=False)
        if feats is None:
            skipped += 1
            features_list.append(None)
        else:
            features_list.append([feats[k] for k in ORIGINAL_FEATURE_COLS])

    print(f"  Extraction complete. {total - skipped} valid, {skipped} skipped.")
    return features_list


def evaluate_detectors(df, max_per_model=2000):
    """Evaluate our existing RF detectors on the frontier dataset."""
    results = []

    # Load detectors
    detector_paths = {
        'all': os.path.join(MODELS_DIR, 'detector_all.joblib'),
        'news': os.path.join(MODELS_DIR, 'detector_news.joblib'),
        'academic': os.path.join(MODELS_DIR, 'detector_academic.joblib'),
        'social': os.path.join(MODELS_DIR, 'detector_social.joblib'),
        'creative': os.path.join(MODELS_DIR, 'detector_creative.joblib'),
    }

    detectors = {}
    for name, path in detector_paths.items():
        if os.path.exists(path):
            d = joblib.load(path)
            detectors[name] = d['model'] if isinstance(d, dict) else d
            print(f"Loaded {name} detector")

    if 'all' not in detectors:
        print("ERROR: all-register detector not found.")
        return pd.DataFrame()

    # Get unique source models excluding human
    source_models = [m for m in df['source_model'].unique() if m != 'Human_Story']
    print(f"\nAI source models in dataset: {source_models}")

    human_df = df[df['source_model'] == 'Human_Story']

    # For each source model, evaluate our detector against human stories
    for source in source_models:
        ai_subset = df[df['source_model'] == source].head(max_per_model)
        h_subset = human_df.head(len(ai_subset))
        
        subset = pd.concat([ai_subset, h_subset]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        if len(subset) < 20:
            print(f"  Skipping {source}: only {len(subset)} samples")
            continue

        # Extract features
        feat_list = extract_features_batch(subset['text_clean'].tolist(), max_samples=len(subset))

        # Filter out None entries
        valid_mask = [f is not None for f in feat_list]
        X = np.array([f for f in feat_list if f is not None])
        y = subset['binary_label'].values[:len(feat_list)][valid_mask]

        if len(X) < 20 or len(np.unique(y)) < 2:
            print(f"  Skipping {source}: insufficient valid samples ({len(X)}) or single class")
            continue

        # Evaluate with each detector
        for det_name, det_model in detectors.items():
            try:
                proba = det_model.predict_proba(X)
                classes = det_model.classes_
                ai_idx = list(classes).index(1) if 1 in classes else 1
                ai_probs = proba[:, ai_idx]

                auc = roc_auc_score(y, ai_probs)
                preds = (ai_probs >= 0.5).astype(int)
                acc = accuracy_score(y, preds)

                results.append({
                    'source_model': source,
                    'detector': det_name,
                    'n_samples': len(X),
                    'auc': round(auc, 4),
                    'accuracy': round(acc, 4),
                    'mean_ai_prob': round(float(np.mean(ai_probs)), 4),
                })
                print(f"  {source} | {det_name} detector | AUC={auc:.4f} | Acc={acc:.4f} | n={len(X)}")
            except Exception as e:
                print(f"  ERROR evaluating {source} with {det_name}: {e}")

    return pd.DataFrame(results)


def main():
    t0 = time.time()

    # Step 1: Download dataset
    df = load_defactify()

    # Step 2: Prepare binary labels
    df = prepare_binary_labels(df)

    # Step 3: Evaluate detectors
    results_df = evaluate_detectors(df, max_per_model=2000)

    # Step 4: Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'frontier_baseline.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Step 5: Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FRONTIER BASELINE EVALUATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    if not results_df.empty:
        # Pivot table: source model × detector AUC
        pivot = results_df.pivot_table(
            index='source_model', columns='detector', values='auc', aggfunc='first'
        )
        print("\nAUC by Source Model × Detector:")
        print(pivot.to_string())

        # Overall summary
        overall = results_df.groupby('detector')['auc'].mean()
        print(f"\nMean AUC across all frontier models:")
        for det, auc in overall.items():
            print(f"  {det}: {auc:.4f}")


if __name__ == '__main__':
    main()
