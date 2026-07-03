#!/usr/bin/env python3
"""
Run Binoculars detector on the same corpus for direct comparison.

Binoculars [Hans et al., 2024] uses two LLMs to compute a score based on
cross-perplexity. This script wraps the implementation for fair comparison
on the same data used by the stylometric detector.

Requirements:
    pip install transformers torch

Usage:
    python scripts/13_binoculars_baseline.py [--max-texts 5000] [--batch-size 16]

Outputs:
  results/binoculars_scores.csv
  results/binoculars_vs_stylometric.csv
"""
import os
import sys
import time
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, normalize_unicode

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run Binoculars on corpus for direct comparison.')
    parser.add_argument('--max-texts', type=int, default=5000,
                        help='Max texts per label (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for Binoculars inference (default: 16)')
    parser.add_argument('--observer-model', type=str, default='huggingface/persimmon-8b-chat',
                        help='Observer model for Binoculars')
    parser.add_argument('--performer-model', type=str, default='huggingface/persimmon-8b-chat',
                        help='Performer model for Binoculars')
    return parser.parse_args()


def compute_binoculars_score(text, observer_tokenizer, performer_tokenizer,
                              observer_model, performer_model, device, max_tokens=512):
    """Compute Binoculars score for a single text.

    Binoculars score = cross_perplexity / observer_perplexity

    Where:
        observer_perplexity = exp(loss(observer_model(text)))
        cross_perplexity = exp(loss(performer_model(text | observer_prefix)))
    """
    import torch

    text = text[:4000]  # Truncate to avoid OOM

    # GPT2 tokenizer has no pad_token by default
    if observer_tokenizer.pad_token is None:
        observer_tokenizer.pad_token = observer_tokenizer.eos_token
    if performer_tokenizer.pad_token is None:
        performer_tokenizer.pad_token = performer_tokenizer.eos_token

    enc = observer_tokenizer(
        text, return_tensors='pt', truncation=True, max_length=max_tokens,
        padding=True, return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        observer_logits = observer_model(**enc).logits
        performer_logits = performer_model(**enc).logits

    # Compute observer perplexity
    shift_logits = observer_logits[..., :-1, :].contiguous()
    shift_labels = enc.input_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    observer_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())

    mask = enc.attention_mask[..., 1:].contiguous()
    observer_loss = (observer_loss * mask).sum() / mask.sum()
    observer_ppl = torch.exp(observer_loss)

    # Compute cross-perplexity
    shift_performer = performer_logits[..., :-1, :].contiguous()
    cross_loss = loss_fct(
        shift_performer.view(-1, shift_performer.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())
    cross_loss = (cross_loss * mask).sum() / mask.sum()
    cross_ppl = torch.exp(cross_loss)

    score = cross_ppl / observer_ppl
    return float(score.item())


def main():
    args = parse_args()

    # Load data — sample from raw corpus, extract features on the fly
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')

    if not os.path.exists(raw_path):
        print("ERROR: Need corpus_raw.parquet")
        return

    # Load only needed columns from raw corpus
    print("Loading corpus (text, label, register only)...")
    df = pd.read_parquet(raw_path, columns=['text', 'label', 'register'])
    df = df.dropna(subset=['text'])

    # Sample
    ai_df = df[df['label'] == 1].sample(min(args.max_texts, len(df[df['label'] == 1])), random_state=42)
    human_df = df[df['label'] == 0].sample(min(args.max_texts, len(df[df['label'] == 0])), random_state=42)
    eval_df = pd.concat([ai_df, human_df]).sample(frac=1, random_state=42)

    print(f"Evaluating on {len(eval_df)} texts ({len(ai_df)} AI, {len(human_df)} human)")

    # Load stylometric model for comparison
    with open(os.path.join(MODELS_DIR, 'manifest.json')) as f:
        manifest = json.load(f)
    _stylo = joblib.load(os.path.join(MODELS_DIR, manifest['all_register_detector']))
    stylometric_model = _stylo['model'] if isinstance(_stylo, dict) else _stylo

    # Compute stylometric scores
    print("\nComputing stylometric scores...")
    stylo_scores = []
    stylo_feats = []
    for _, row in eval_df.iterrows():
        fv = extract_feature_vector(row['text'], feature_cols=FEATURE_COLS, extended=False)
        if fv is not None:
            stylo_feats.append(fv)
        else:
            stylo_feats.append([0.0] * len(FEATURE_COLS))

    stylo_proba = stylometric_model.predict_proba(np.array(stylo_feats))[:, 1]
    y_true = eval_df['label'].values
    stylo_auc = roc_auc_score(y_true, stylo_proba)
    print(f"Stylometric AUC: {stylo_auc:.4f}")

    # Try to load Binoculars models
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\nLoading Binoculars models...")
        print(f"  Observer: {args.observer_model}")
        print(f"  Performer: {args.performer_model}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        observer_tokenizer = AutoTokenizer.from_pretrained(args.observer_model)
        performer_tokenizer = AutoTokenizer.from_pretrained(args.performer_model)
        observer_model = AutoModelForCausalLM.from_pretrained(
            args.observer_model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device)
        performer_model = AutoModelForCausalLM.from_pretrained(
            args.performer_model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        ).to(device)
        observer_model.eval()
        performer_model.eval()

        # Compute Binoculars scores
        print(f"\nComputing Binoculars scores (batch_size={args.batch_size})...")
        bino_scores = []
        t0 = time.time()

        for i, (_, row) in enumerate(eval_df.iterrows()):
            text = normalize_unicode(row['text'])
            score = compute_binoculars_score(
                text, observer_tokenizer, performer_tokenizer,
                observer_model, performer_model, device,
            )
            bino_scores.append(score)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(eval_df) - i - 1) / rate
                print(f"  {i+1}/{len(eval_df)} ({rate:.1f} texts/sec, ETA: {eta:.0f}s)")

        bino_scores = np.array(bino_scores)
        # Binoculars: lower score = more likely AI (AI text has lower perplexity ratio)
        # So we negate for AUC computation (higher = AI)
        bino_auc = roc_auc_score(y_true, -bino_scores)
        bino_acc = accuracy_score(y_true, (-bino_scores > np.median(-bino_scores)).astype(int))
        print(f"\nBinoculars AUC: {bino_auc:.4f}")

        # Save scores
        scores_df = pd.DataFrame({
            'label': y_true,
            'register': eval_df['register'].values,
            'stylometric_score': stylo_proba,
            'binoculars_score': bino_scores,
            'binoculars_score_neg': -bino_scores,
        })
        scores_path = os.path.join(RESULTS_DIR, 'binoculars_scores.csv')
        scores_df.to_csv(scores_path, index=False)
        print(f"Saved scores to {scores_path}")

        # Comparison table
        comp_rows = [
            {'method': 'stylometric', 'auc': stylo_auc, 'n_texts': len(eval_df)},
            {'method': 'binoculars', 'auc': bino_auc, 'n_texts': len(eval_df)},
        ]

        # Simple ensemble: average of normalized scores
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        stylo_norm = scaler.fit_transform(stylo_proba.reshape(-1, 1)).ravel()
        bino_norm = scaler.fit_transform((-bino_scores).reshape(-1, 1)).ravel()
        ensemble_score = (stylo_norm + bino_norm) / 2
        ensemble_auc = roc_auc_score(y_true, ensemble_score)
        comp_rows.append({'method': 'ensemble_avg', 'auc': ensemble_auc, 'n_texts': len(eval_df)})

        comp_df = pd.DataFrame(comp_rows)
        comp_path = os.path.join(RESULTS_DIR, 'binoculars_vs_stylometric.csv')
        comp_df.to_csv(comp_path, index=False)
        print(f"Saved comparison to {comp_path}")

        print("\n" + "=" * 50)
        print(f"{'Method':<20} {'AUC':>8}")
        print("-" * 50)
        for _, row in comp_df.iterrows():
            print(f"{row['method']:<20} {row['auc']:>8.4f}")

    except ImportError as e:
        print(f"\nERROR: Required packages not installed: {e}")
        print("Install with: pip install transformers torch")
        print("\nSaving stylometric-only scores for later comparison...")

        scores_df = pd.DataFrame({
            'label': y_true,
            'register': eval_df['register'].values,
            'stylometric_score': stylo_proba,
        })
        scores_path = os.path.join(RESULTS_DIR, 'binoculars_scores.csv')
        scores_df.to_csv(scores_path, index=False)
        print(f"Saved to {scores_path}")

    except Exception as e:
        print(f"\nERROR running Binoculars: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
