#!/usr/bin/env python3
"""Fast SOTA evaluation: MAGE Longformer on MAGE and TuringBench."""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from huggingface_hub import snapshot_download

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_DIR)


def _get_device():
    return 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def load_detector(model_name):
    print(f"Loading {model_name}...")
    cache_dir = os.path.join(PROJECT_DIR, 'models', 'hf_cache', model_name.replace('/', '--'))
    local_dir = os.path.join(PROJECT_DIR, 'models', 'public', model_name.replace('/', '--'))
    if not os.path.exists(local_dir):
        print(f"  Downloading snapshot to {local_dir}...")
        snapshot_download(repo_id=model_name, local_dir=local_dir, cache_dir=cache_dir)
    config_path = os.path.join(local_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    # Remove broken id2label/label2id if they cause validation errors
    config_dict.pop('id2label', None)
    config_dict.pop('label2id', None)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    # Backup original if needed
    orig_config = config_path + '.orig'
    if not os.path.exists(orig_config):
        shutil.copy(config_path, orig_config)
    config = AutoConfig.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir, config=config)
    device = _get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def batch_probs(texts, tokenizer, model, device, max_length=4096, batch_size=8, ai_label=1):
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if i > 0 and i % 200 == 0:
            print(f"  inferred {i}/{len(texts)}...")
        try:
            inputs = tokenizer(list(batch), return_tensors='pt', truncation=True, max_length=max_length, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
                p = torch.softmax(logits, dim=-1)[:, ai_label].cpu().numpy()
                probs.extend([float(x) for x in p])
        except Exception as e:
            print(f"Detector error at {i}: {e}")
            probs.extend([0.5] * len(batch))
    return probs


def load_mage():
    try:
        mage = load_dataset("yaful/MAGE")
        df = mage['test'].to_pandas()
        df['label'] = 1 - df['label'].astype(int)
        return df
    except Exception as e:
        print(f"Could not load MAGE: {e}")
        return None


def load_turingbench():
    try:
        tb = load_dataset("turingbench/TuringBench", revision="refs/convert/parquet")
        df = tb['train'].to_pandas()
        df['label'] = df['label'].apply(lambda x: 0 if x == 'human' else 1)
        df['text'] = df['Generation']
        return df
    except Exception as e:
        print(f"Could not load TuringBench: {e}")
        return None


def evaluate_model(df, model_name, max_length, ai_label=1, max_total=2000, batch_size=8):
    print(f"\n=== {model_name} @ {max_length} tokens ===")
    sample = df.sample(n=min(max_total, len(df)), random_state=42)
    texts = sample['text'].tolist()
    labels = sample['label'].values

    tokenizer, model, device = load_detector(model_name)
    probs = batch_probs(texts, tokenizer, model, device, max_length=max_length, batch_size=batch_size, ai_label=ai_label)
    del model, tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, [1 if p >= 0.5 else 0 for p in probs])
    print(f"AUC: {auc:.4f}, Acc: {acc:.4f} (n={len(labels)})")
    return {'model': model_name, 'max_length': max_length, 'auc': auc, 'accuracy': acc, 'n_samples': len(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=200, help='Samples per benchmark')
    parser.add_argument('--max_length', type=int, default=2048, help='Max tokens')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--model', type=str, default='nealcly/detection-longformer', help='HuggingFace model name')
    parser.add_argument('--ai_label', type=int, default=0, help='Logit index for AI class')
    args = parser.parse_args()

    results = []

    mage_df = load_mage()
    if mage_df is not None:
        results.append(evaluate_model(mage_df, args.model, args.max_length, ai_label=args.ai_label, max_total=args.n_samples, batch_size=args.batch_size))

    tb_df = load_turingbench()
    if tb_df is not None:
        results.append(evaluate_model(tb_df, args.model, args.max_length, ai_label=args.ai_label, max_total=args.n_samples, batch_size=args.batch_size))

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    out_path = os.path.join(PROJECT_DIR, 'results', 'fast_sota_eval.csv')
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
