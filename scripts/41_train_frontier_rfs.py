#!/usr/bin/env python3
"""
Phase 1C: Retrain stylometric Random Forest detectors on a mixed corpus
combining the old benchmark data and the new Defactify frontier dataset.

Uses multiprocessing to extract features in parallel.
Predicts registers for Defactify texts using the pre-trained register classifier.
Outputs:
    Overwrites models/detector_*.joblib (creating backups)
    Prints evaluation table on the held-out Defactify test split
"""
import os
import sys
import time
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import Parallel, delayed

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from tool.feature_extractor import extract_features, ORIGINAL_FEATURE_COLS

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy'
]


def extract_single_text(text):
    """Helper to extract features from a single text."""
    feats = extract_features(text, extended=False)
    if feats is None:
        return None
    n_words = len(text.split())
    n_sents = max(1, text.count('.') + text.count('?') + text.count('!'))
    return [feats[k] for k in ORIGINAL_FEATURE_COLS] + [n_words, n_sents]


def extract_features_parallel(texts, n_jobs=-1):
    """Extract features in parallel using joblib."""
    print(f"Extracting features for {len(texts)} texts in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_single_text)(t) for t in texts
    )
    # Filter out None and split features and valid indices
    valid_indices = []
    valid_features = []
    for idx, res in enumerate(results):
        if res is not None:
            valid_indices.append(idx)
            valid_features.append(res)
    return np.array(valid_features), valid_indices


def main():
    t0 = time.time()

    # Step 1: Load old corpus features
    old_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    print(f"Loading old corpus features from {old_path}...")
    df_old = pd.read_parquet(old_path)
    # Ensure correct columns
    df_old = df_old[FEATURE_COLS + ['label', 'register', 'model', 'source']]
    print(f"Old corpus size: {len(df_old)}")

    # Step 2: Download/Load Defactify dataset
    from datasets import load_dataset
    print("Loading Defactify dataset...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset")
    df_train = ds['train'].to_pandas()
    df_test = ds['test'].to_pandas() if 'test' in ds else ds['validation'].to_pandas()

    print(f"Defactify loaded: train={len(df_train)}, test={len(df_test)}")

    # Prepare datasets
    for df in (df_train, df_test):
        df['binary_label'] = df['Label_A'].astype(int)
        df['text_clean'] = df['Text'].astype(str)
        df['source_model'] = df['Label_B'].astype(str)

    # Step 3: Sample for training
    # For train: take up to 3000 samples per AI model and a balanced set of human stories
    max_train_per_model = 3000
    ai_models = [m for m in df_train['source_model'].unique() if m != 'Human_Story']
    human_stories = df_train[df_train['source_model'] == 'Human_Story']

    train_parts = []
    for model in ai_models:
        subset = df_train[df_train['source_model'] == model].head(max_train_per_model)
        train_parts.append(subset)
    
    # Balance with equal number of total human stories
    total_ai_train = sum(len(x) for x in train_parts)
    h_train = human_stories.sample(min(total_ai_train, len(human_stories)), random_state=42)
    train_parts.append(h_train)
    df_train_sampled = pd.concat(train_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Sampled train set size: {len(df_train_sampled)}")

    # Sample for testing
    max_test_per_model = 500
    test_parts = []
    for model in ai_models:
        subset = df_test[df_test['source_model'] == model].head(max_test_per_model)
        test_parts.append(subset)
    
    h_test = df_test[df_test['source_model'] == 'Human_Story'].sample(
        min(sum(len(x) for x in test_parts), len(df_test[df_test['source_model'] == 'Human_Story'])),
        random_state=42
    )
    test_parts.append(h_test)
    df_test_sampled = pd.concat(test_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Sampled test set size: {len(df_test_sampled)}")

    # Step 4: Extract features in parallel
    print("\n--- Extracting Train Features ---")
    X_train_raw, train_valid_idx = extract_features_parallel(df_train_sampled['text_clean'].tolist())
    df_train_feats = pd.DataFrame(X_train_raw, columns=FEATURE_COLS + ['n_words', 'n_sents'])
    df_train_meta = df_train_sampled.iloc[train_valid_idx].reset_index(drop=True)
    df_train_final = pd.concat([df_train_feats, df_train_meta[['binary_label', 'source_model']]], axis=1)
    df_train_final.rename(columns={'binary_label': 'label', 'source_model': 'model'}, inplace=True)
    df_train_final['source'] = 'defactify_train'

    print("\n--- Extracting Test Features ---")
    X_test_raw, test_valid_idx = extract_features_parallel(df_test_sampled['text_clean'].tolist())
    df_test_feats = pd.DataFrame(X_test_raw, columns=FEATURE_COLS + ['n_words', 'n_sents'])
    df_test_meta = df_test_sampled.iloc[test_valid_idx].reset_index(drop=True)
    df_test_final = pd.concat([df_test_feats, df_test_meta[['binary_label', 'source_model']]], axis=1)
    df_test_final.rename(columns={'binary_label': 'label', 'source_model': 'model'}, inplace=True)
    df_test_final['source'] = 'defactify_test'

    # Step 5: Predict registers for Defactify samples
    print("\nPredicting registers for Defactify data...")
    rc_data = joblib.load(os.path.join(MODELS_DIR, 'register_classifier.joblib'))
    reg_clf = rc_data['model']
    le = rc_data['label_encoder']

    # Predict train registers
    train_reg_preds = reg_clf.predict(df_train_final[FEATURE_COLS].values)
    df_train_final['register'] = le.inverse_transform(train_reg_preds)

    # Predict test registers
    test_reg_preds = reg_clf.predict(df_test_final[FEATURE_COLS].values)
    df_test_final['register'] = le.inverse_transform(test_reg_preds)

    print(f"Train register distribution:\n{df_train_final['register'].value_counts()}")

    # Step 6: Combine with old corpus features
    # Subsample old corpus to keep registers balanced (max 30,000 per register)
    max_old_per_register = 30000
    old_parts = []
    for reg, grp in df_old.groupby('register'):
        old_parts.append(grp.sample(min(max_old_per_register, len(grp)), random_state=42))
    df_old_sampled = pd.concat(old_parts)

    df_combined_train = pd.concat([df_old_sampled, df_train_final], ignore_index=True)
    print(f"\nCombined training corpus size: {len(df_combined_train)}")

    # Step 7: Train the Random Forests!
    print("\n=== Training Retrained Random Forest Models ===")
    
    # 7A. Train detector_all
    print("Training detector_all...")
    clf_all = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_all.fit(df_combined_train[FEATURE_COLS].values, df_combined_train['label'].values)
    
    # Backup and save
    all_path = os.path.join(MODELS_DIR, 'detector_all.joblib')
    if os.path.exists(all_path):
        shutil.copy(all_path, all_path + '.bak')
    joblib.dump({'model': clf_all, 'feature_cols': FEATURE_COLS, 'register': 'all'}, all_path)
    print(f"Saved detector_all.joblib (backup created)")

    # 7B. Train per-register detectors
    for reg in ['academic', 'news', 'social', 'creative']:
        print(f"Training detector_{reg}...")
        sub_train = df_combined_train[df_combined_train['register'] == reg]
        if len(sub_train) < 100 or sub_train['label'].nunique() < 2:
            print(f"  Skipping {reg}: insufficient combined training data ({len(sub_train)})")
            continue
            
        clf_reg = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_reg.fit(sub_train[FEATURE_COLS].values, sub_train['label'].values)
        
        reg_path = os.path.join(MODELS_DIR, f'detector_{reg}.joblib')
        if os.path.exists(reg_path):
            shutil.copy(reg_path, reg_path + '.bak')
        joblib.dump({'model': clf_reg, 'feature_cols': FEATURE_COLS, 'register': reg}, reg_path)
        print(f"Saved detector_{reg}.joblib (backup created)")

    # Step 8: Evaluate retrained models on held-out Defactify test split
    print("\n=== Evaluating Retrained Models on Frontier Test Split ===")
    
    # Load detectors
    detectors = {
        'all': clf_all
    }
    for reg in ['academic', 'news', 'social', 'creative']:
        reg_path = os.path.join(MODELS_DIR, f'detector_{reg}.joblib')
        if os.path.exists(reg_path):
            detectors[reg] = joblib.load(reg_path)['model']

    eval_results = []
    
    # Group test set by the generator model
    test_models = [m for m in df_test_final['model'].unique() if m != 'Human_Story']
    human_test = df_test_final[df_test_final['model'] == 'Human_Story']

    for model in test_models:
        ai_test = df_test_final[df_test_final['model'] == model]
        # Pair with equal human samples
        sub_test = pd.concat([ai_test, human_test.sample(min(len(ai_test), len(human_test)), random_state=42)])
        
        X_test = sub_test[FEATURE_COLS].values
        y_test = sub_test['label'].values

        for det_name, det_model in detectors.items():
            proba = det_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            preds = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, preds)

            eval_results.append({
                'source_model': model,
                'detector': det_name,
                'auc': round(auc, 4),
                'accuracy': round(acc, 4)
            })
            print(f"  {model} | {det_name} detector | AUC={auc:.4f} | Acc={acc:.4f}")

    eval_df = pd.DataFrame(eval_results)
    out_path = os.path.join(RESULTS_DIR, 'frontier_retrained_evaluation.csv')
    eval_df.to_csv(out_path, index=False)
    print(f"\nSaved retrained evaluation results to {out_path}")

    # Print summary pivot
    pivot = eval_df.pivot_table(index='source_model', columns='detector', values='auc')
    print("\nAUC of Retrained RFs by Source Model × Detector:")
    print(pivot.to_string())

    print(f"\nTotal elapsed time: {time.time() - t0:.0f} seconds")


if __name__ == '__main__':
    main()
