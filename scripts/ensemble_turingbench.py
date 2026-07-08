#!/usr/bin/env python3
"""Train a logistic regression ensemble over multiple fine-tuned TuringBench detectors."""
import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ensemble over TuringBench detectors")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True,
                        help="Paths to fine-tuned model directories")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "models", "turingbench_ensemble"),
                        help="Directory to save ensemble wrapper")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Token max length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Inference batch size per model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model_and_tokenizer(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_ai_probabilities(model, tokenizer, texts, device, max_length, batch_size):
    """Return AI-class probabilities for a list of texts."""
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(batch_probs.tolist())
    return np.array(probs)


def load_turingbench_val(max_val_samples=None):
    print("Loading TuringBench validation split...")
    tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
    val = tb['validation'].to_pandas()
    val['label'] = val['label'].apply(lambda x: 0 if str(x).lower() == 'human' else 1)
    val['text'] = val['Generation'].astype(str)
    val = val[val['text'].str.len() >= 20].reset_index(drop=True)
    if max_val_samples is not None:
        val = val.sample(n=min(max_val_samples, len(val)), random_state=42).reset_index(drop=True)
    print(f"Validation rows: {len(val)}")
    return val


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    val_df = load_turingbench_val()
    texts = val_df['text'].tolist()
    labels = val_df['label'].values

    # Collect probabilities from each model
    model_names = []
    prob_matrix = []

    for model_dir in args.model_dirs:
        if not os.path.isdir(model_dir):
            print(f"WARNING: {model_dir} not found, skipping.")
            continue
        name = os.path.basename(os.path.normpath(model_dir))
        print(f"\nLoading model: {name}")
        tokenizer, model = load_model_and_tokenizer(model_dir, device)
        probs = get_ai_probabilities(
            model, tokenizer, texts, device, args.max_length, args.batch_size
        )
        model_names.append(name)
        prob_matrix.append(probs)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, (probs >= 0.5).astype(int))
        print(f"  {name} single-model AUC: {auc:.4f}, accuracy: {acc:.4f}")
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    if len(model_names) < 2:
        print("Need at least 2 models to train an ensemble. Exiting.")
        return

    X = np.column_stack(prob_matrix)

    # Train logistic regression ensemble on validation probabilities
    print("\nTraining logistic regression ensemble...")
    ensemble = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    ensemble.fit(X, labels)

    ensemble_probs = ensemble.predict_proba(X)[:, 1]
    ensemble_auc = roc_auc_score(labels, ensemble_probs)
    ensemble_acc = accuracy_score(labels, (ensemble_probs >= 0.5).astype(int))
    print(f"\nEnsemble AUC: {ensemble_auc:.4f}")
    print(f"Ensemble accuracy: {ensemble_acc:.4f}")

    # Save ensemble
    ensemble_path = os.path.join(args.output_dir, 'ensemble.joblib')
    joblib.dump({
        'model': ensemble,
        'model_names': model_names,
        'model_dirs': args.model_dirs,
        'max_length': args.max_length,
    }, ensemble_path)
    print(f"Saved ensemble to {ensemble_path}")

    # Save results
    results = pd.DataFrame([{
        'models': ', '.join(model_names),
        'num_models': len(model_names),
        'max_length': args.max_length,
        'val_auc': ensemble_auc,
        'val_accuracy': ensemble_acc,
    }])
    results_path = os.path.join(PROJECT_DIR, 'results', 'turingbench_ensemble.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")


if __name__ == '__main__':
    main()
