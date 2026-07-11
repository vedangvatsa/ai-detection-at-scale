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
    
    print("\nEvaluating 5 human samples from Beemo:")
    count = 0
    for _, row in df.iterrows():
        h = row['human_output']
        if isinstance(h, str) and len(h.strip()) >= 50:
            inputs = tokenizer(h, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits.numpy()[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            print(f"\nText snippet: {h[:100]}...")
            print(f"Tokens: {inputs.input_ids[0].tolist()[:20]}...")
            print(f"Logits: {[round(float(l), 4) for l in logits]}, Probs: {[round(float(p), 4) for p in probs]}")
            count += 1
            if count >= 5:
                break

if __name__ == '__main__':
    main()
