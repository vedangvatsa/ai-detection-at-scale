#!/usr/bin/env python3
"""
Error analysis: sample false positives and false negatives from the 31-feature model.
Shows examples of human text flagged as AI and AI text flagged as human.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from tool.feature_extractor import ALL_FEATURE_COLS

RANDOM_SEED = 42


def main():
    # Load 31-feature model
    model_path = os.path.join(MODELS_DIR, 'detector_all_31.joblib')
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found. Run script 11 first.")
        return

    model = joblib.load(model_path)
    print(f"Loaded 31-feature model from {model_path}")

    # Load 31-feature corpus (only needed columns)
    feat_path = os.path.join(DATA_DIR, 'corpus_features_31.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path, columns=['label', 'register'] + ALL_FEATURE_COLS)
    df = df.dropna(subset=ALL_FEATURE_COLS)
    print(f"Loaded {len(df)} texts from features corpus")

    # Predict scores
    X = df[ALL_FEATURE_COLS].values
    y = df['label'].values
    y_proba = model.predict_proba(X)[:, 1]

    # Add predictions
    df['score'] = y_proba
    df['pred'] = (y_proba >= 0.5).astype(int)

    # Overall metrics
    auc = roc_auc_score(y, y_proba)
    acc = (df['pred'] == y).mean()
    print(f"\nOverall: AUC={auc:.4f}, Accuracy={acc:.4f}")

    # Find errors
    fps = df[(y == 0) & (df['pred'] == 1)]  # human flagged as AI
    fns = df[(y == 1) & (df['pred'] == 0)]  # AI flagged as human

    print(f"\nFalse Positives (human → AI): {len(fps)}")
    print(f"False Negatives (AI → human): {len(fns)}")

    # Sample error indices per register (avoid loading all raw text)
    error_idx = []
    for register in ['academic', 'news', 'social', 'creative']:
        for error_type, df_err in [('FP', fps), ('FN', fns)]:
            df_reg = df_err[df_err['register'] == register]
            if len(df_reg) == 0:
                continue
            n = min(3, len(df_reg))
            sample = df_reg.sample(n, random_state=RANDOM_SEED)
            for idx, row in sample.iterrows():
                error_idx.append({
                    'index': idx,
                    'register': register,
                    'error_type': error_type,
                    'score': row['score']
                })

    # Load only the sampled text rows from raw corpus
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found.")
        return

    print(f"\nLoading text for {len(error_idx)} sampled errors...")
    idx_list = [x['index'] for x in error_idx]
    # Read only the text column and select the sampled indices
    text_df = pd.read_parquet(raw_path, columns=['text'])
    text_df = text_df.loc[text_df.index.isin(idx_list)]
    text_df = text_df.reindex(idx_list)

    # Build examples
    error_examples = []
    for item in error_idx:
        text = text_df.loc[item['index'], 'text']
        if not isinstance(text, str):
            text = str(text)
        error_examples.append({
            'register': item['register'],
            'error_type': item['error_type'],
            'score': item['score'],
            'text': text[:400] + '...' if len(text) > 400 else text
        })

    # Save error examples
    err_df = pd.DataFrame(error_examples)
    err_path = os.path.join(RESULTS_DIR, 'error_examples.csv')
    err_df.to_csv(err_path, index=False)
    print(f"\nSaved {len(err_df)} error examples to {err_path}")

    # Show examples
    print("\n=== Error Examples ===")
    for _, row in err_df.head(12).iterrows():
        print(f"\n[{row['register']} | {row['error_type']} | score={row['score']:.3f}]")
        print(row['text'])

    # Error rate by register
    print("\n=== Error Rates by Register ===")
    rates = []
    for register in ['academic', 'news', 'social', 'creative']:
        df_reg = df[df['register'] == register]
        if len(df_reg) == 0:
            continue
        fp_rate = (fps[fps['register'] == register].shape[0] / df_reg[df_reg['label'] == 0].shape[0]) if (df_reg['label'] == 0).sum() > 0 else np.nan
        fn_rate = (fns[fns['register'] == register].shape[0] / df_reg[df_reg['label'] == 1].shape[0]) if (df_reg['label'] == 1).sum() > 0 else np.nan
        auc_reg = roc_auc_score(df_reg['label'], df_reg['score'])
        rates.append({
            'register': register,
            'n': len(df_reg),
            'auc': auc_reg,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate
        })
        print(f"{register:10s} n={len(df_reg):6d} AUC={auc_reg:.3f} FP={fp_rate:.3f} FN={fn_rate:.3f}")

    rates_df = pd.DataFrame(rates)
    rates_path = os.path.join(RESULTS_DIR, 'error_rates_by_register.csv')
    rates_df.to_csv(rates_path, index=False)
    print(f"\nSaved error rates to {rates_path}")


if __name__ == '__main__':
    main()
