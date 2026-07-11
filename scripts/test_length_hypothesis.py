#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    model_id = "vedangvatsa/vedang-turingbench-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    
    # 1. Very short human text
    short_human = "Trieste is located at the head of the Gulf of Trieste."
    
    # 2. Repeated short human text (to make it long)
    long_human = " ".join([short_human] * 15)
    
    # 3. Real news article human text (from CNN or similar, very long)
    news_human = """
    The Federal Reserve on Wednesday raised its benchmark interest rate by three-quarters of a percentage point
    for a second straight time in its most aggressive effort in three decades to tame inflation. The hike,
    which was widely expected by economists and investors, brings the central bank's key rate to a range of
    2.25% to 2.50%. Fed Chair Jerome Powell said at a press conference that another "unusually large increase"
    could be appropriate at the next meeting in September, but that a decision would depend on economic data
    released between now and then.
    """.strip()
    
    for label, text in [("Short Human", short_human), ("Long Human (Repeated)", long_human), ("News Human (Long)", news_human)]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.numpy()[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        print(f"\n{label}:")
        print(f"Length (words): {len(text.split())}")
        print(f"Logits: {[round(float(l), 4) for l in logits]}, Probs: {[round(float(p), 4) for p in probs]}")

if __name__ == '__main__':
    main()
