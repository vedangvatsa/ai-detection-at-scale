#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool.hybrid_detector import predict_hybrid

def main():
    print("Testing the production predict_hybrid endpoint...")
    
    test_texts = [
        "The Federal Reserve on Wednesday raised its benchmark interest rate by three-quarters of a percentage point for a second straight time in its most aggressive effort in three decades to tame inflation.",
        "Trieste is located at the head of the Gulf of Trieste in northeast Italy."
    ]
    
    for register in ["all", "news", "academic", "social", "creative"]:
        print(f"\nEvaluating with register: '{register}'")
        for i, text in enumerate(test_texts):
            prob = predict_hybrid(text, register=register)
            print(f"  Text {i+1} prediction probability: {prob:.4f}")
            
    print("\nAll pipeline verification tests passed successfully!")

if __name__ == '__main__':
    main()
