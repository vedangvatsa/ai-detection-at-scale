#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    model_id = "vedangvatsa/vedang-turingbench-roberta-large"
    print(f"Loading PyTorch model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    
    print("Loading TuringBench dataset...")
    # TuringBench might be under 'turingbench' or another dataset ID. Let's try downloading 'turingbench'
    try:
        dataset = load_dataset("turingbench/TuringBench", trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load 'turingbench' from HF: {e}")
        # Try loading a local file if it exists, or just query Hugging Face datasets search
        return
        
    print("TuringBench loaded successfully!")
    print(dataset)
    
    # We will test on the test set if available, otherwise train set
    split_name = 'test' if 'test' in dataset else 'train'
    df = dataset[split_name].to_pandas()
    
    # TuringBench has multiple classes. Let's inspect the columns
    print("Columns:", df.columns)
    print("Sample labels count:\n", df['label'].value_counts()[:10])
    
    # In TuringBench: label 0 is human?
    # Let's check!
    # Let's evaluate a few samples for each label
    unique_labels = df['label'].unique()
    for lbl in sorted(unique_labels)[:5]:
        subset = df[df['label'] == lbl].head(3)
        print(f"\n--- Testing Label: {lbl} ---")
        for idx, row in subset.iterrows():
            text = row['Generation']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.numpy()[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            print(f"Logits: {[round(float(l), 4) for l in logits]}, Probs: {[round(float(p), 4) for p in probs]}")

if __name__ == '__main__':
    main()
