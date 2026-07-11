#!/usr/bin/env python3
import os
import sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    tb_id = "vedangvatsa/vedang-turingbench-roberta-large"
    std_id = "roberta-large"
    
    print(f"Loading TuringBench tokenizer from {tb_id}...")
    tokenizer_tb = AutoTokenizer.from_pretrained(tb_id)
    print(f"TuringBench Tokenizer vocab size: {tokenizer_tb.vocab_size if hasattr(tokenizer_tb, 'vocab_size') else 'N/A'}")
    
    print(f"Loading standard tokenizer from {std_id}...")
    tokenizer_std = AutoTokenizer.from_pretrained(std_id)
    print(f"Standard Tokenizer vocab size: {tokenizer_std.vocab_size if hasattr(tokenizer_std, 'vocab_size') else 'N/A'}")
    
    t = "Hello world! This is a test sentence."
    
    inputs_tb = tokenizer_tb(t)
    inputs_std = tokenizer_std(t)
    
    print(f"\nText: {t}")
    print(f"TuringBench token IDs: {inputs_tb['input_ids']}")
    print(f"Standard token IDs:    {inputs_std['input_ids']}")

if __name__ == '__main__':
    main()
