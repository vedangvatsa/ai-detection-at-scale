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
    tokenizer_id = "roberta-large"
    
    print(f"Loading PyTorch model from {model_id}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    
    print(f"Loading tokenizer from {tokenizer_id}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    print("Loading TuringBench dataset...")
    try:
        dataset = load_dataset("turingbench/TuringBench")
    except Exception as e:
        print(f"Failed: {e}")
        return
        
    df = dataset['train'].to_pandas()
    
    # Let's test on actual human vs AI texts from TuringBench
    # Human label is typically 'human' or 0?
    # Let's print unique labels
    print("Unique labels in TuringBench train:", df['label'].unique())
    
    # Let's find some samples of 'human' (which is the human class)
    # and some AI classes
    human_samples = df[df['label'] == 'human'].head(5)
    ai_samples = df[df['label'] != 'human'].head(5)
    
    print("\nEvaluating human samples from TuringBench:")
    for idx, row in human_samples.iterrows():
        text = row['Generation']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.numpy()[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        print(f"Text snippet: {text[:60]}...")
        print(f"Logits: {[round(float(l), 4) for l in logits]}, Probs: {[round(float(p), 4) for p in probs]}")
        
    print("\nEvaluating AI samples from TuringBench:")
    for idx, row in ai_samples.iterrows():
        text = row['Generation']
        label = row['label']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.numpy()[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        print(f"Label: {label}, Text snippet: {text[:60]}...")
        print(f"Logits: {[round(float(l), 4) for l in logits]}, Probs: {[round(float(p), 4) for p in probs]}")

if __name__ == '__main__':
    main()
