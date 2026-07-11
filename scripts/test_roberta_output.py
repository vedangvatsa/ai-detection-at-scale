#!/usr/bin/env python3
import os
import sys
import numpy as np
from datasets import load_dataset
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
ONNX_PATH = os.path.join(MODELS_DIR, 'roberta_large_onnx_quantized.onnx')

def main():
    ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    pytorch_path = os.path.join(MODELS_DIR, 'roberta_large_semantic_model')
    tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
    
    beemo = load_dataset("toloka/beemo")
    df = beemo['train'].to_pandas()
    
    print("Evaluating 20 human samples and 20 AI samples...")
    human_probs = []
    ai_probs = []
    
    # Human samples
    count = 0
    for _, row in df.iterrows():
        h = row['human_output']
        if isinstance(h, str) and len(h.strip()) >= 50:
            inputs = tokenizer(h, return_tensors="np", truncation=True, max_length=256, padding="max_length")
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            logits = ort_session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            prob = float(probs[0][1])
            human_probs.append(prob)
            count += 1
            if count >= 20:
                break
                
    # AI samples
    count = 0
    for _, row in df.iterrows():
        m = row['model_output']
        if isinstance(m, str) and len(m.strip()) >= 50:
            inputs = tokenizer(m, return_tensors="np", truncation=True, max_length=256, padding="max_length")
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            logits = ort_session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            prob = float(probs[0][1])
            ai_probs.append(prob)
            count += 1
            if count >= 20:
                break
                
    print("\nHuman probabilities (expect close to 0):")
    print([round(p, 4) for p in human_probs])
    print(f"Mean Human Prob: {np.mean(human_probs):.4f}")
    
    print("\nAI probabilities (expect close to 1):")
    print([round(p, 4) for p in ai_probs])
    print(f"Mean AI Prob: {np.mean(ai_probs):.4f}")

if __name__ == '__main__':
    main()
