#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool.attribution import attribute_source
from tool.feature_extractor import extract_feature_vector

def main():
    print("Testing Reasoning Model Source Attribution...")
    
    reasoning_text = (
        "To find the sum of all prime numbers less than 100, we should execute the following steps. "
        "First, we will establish a boolean array of size 100 initialized to true. "
        "Second, we will iterate from two up to the square root of 100. "
        "Third, for each prime number found, we will mark all of its multiples as false. "
        "Finally, we will sum the remaining indices that remain true. "
        "This algorithm is known as the Sieve of Eratosthenes and is highly efficient."
    )
    
    # Extract feature vector
    from tool.api import get_models
    m = get_models()
    feat_vector = extract_feature_vector(reasoning_text, feature_cols=m['feature_cols'], extended=False)
    
    # Call attribute_source with text
    res = attribute_source(feat_vector, is_ai_probability=0.95, text=reasoning_text)
    
    print(f"\nAttributed Source: {res['source_model']}")
    print(f"Confidence: {res['confidence']:.4f}")
    
    if "Reasoning LLM" in res['source_model']:
        print("\nSuccess! The text was correctly classified and attributed to a Reasoning Model (o1/DeepSeek-R1).")
    else:
        print("\nFailed: The text was not attributed to a Reasoning Model.")

if __name__ == '__main__':
    main()
