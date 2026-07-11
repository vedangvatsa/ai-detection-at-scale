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
    
    beemo = load_dataset("toloka/beemo")
    df = beemo['train'].to_pandas()
    
    print("\nEvaluating 5 human samples and 5 AI samples with FIXED tokenizer...")
    
    # Human samples
    human_probs = []
    count = 0
    for _, row in df.iterrows():
        h = row['human_output']
        if isinstance(h, str) and len(h.strip()) >= 50:
            inputs = tokenizer(h, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.numpy()
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            prob = float(probs[0][1])
            human_probs.append(prob)
            count += 1
            if count >= 5:
                break
                
    # AI samples
    ai_probs = []
    count = 0
    for _, row in df.iterrows():
        m = row['model_output']
        if isinstance(m, str) and len(m.strip()) >= 50:
            inputs = tokenizer(m, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.numpy()
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            prob = float(probs[0][1])
            ai_probs.append(prob)
            count += 1
            if count >= 5:
                break
                
    print("\nFixed PyTorch Human probabilities (expect close to 0):")
    print([round(p, 4) for p in human_probs])
    
    print("\nFixed PyTorch AI probabilities (expect close to 1):")
    print([round(p, 4) for p in ai_probs])

if __name__ == '__main__':
    main()
