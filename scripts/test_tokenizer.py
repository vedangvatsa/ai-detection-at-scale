#!/usr/bin/env python3
import os
import sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    model_id = "vedangvatsa/vedang-turingbench-roberta-large"
    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    texts = [
        "Hello world!",
        "This is a test sentence that should have different tokens than the first one.",
        "Okay so I was talking to my friend yesterday and we got into this whole thing."
    ]
    
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt")
        print(f"\nText: {t}")
        print(f"Input IDs: {inputs.input_ids[0].tolist()}")
        print(f"Attention Mask: {inputs.attention_mask[0].tolist()}")

if __name__ == '__main__':
    main()
