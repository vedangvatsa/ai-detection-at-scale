#!/usr/bin/env python3
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.")
        return
        
    device = torch.device("mps")
    print(f"Using device: {device}")
    
    model_id = "vedangvatsa/vedang-turingbench-roberta-large"
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    
    # Dummy batch
    text = ["This is a test sentence for benchmarking MPS speed."] * 8
    inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    labels = torch.tensor([1] * 8).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    print("Benchmarking 10 steps...")
    t0 = time.time()
    for i in range(10):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}/10 loss: {loss.item():.4f}")
        
    elapsed = time.time() - t0
    print(f"Total time for 10 steps: {elapsed:.2f} seconds ({elapsed/10:.3f} s/step)")

if __name__ == '__main__':
    main()
