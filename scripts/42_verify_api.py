#!/usr/bin/env python3
"""
Verify the hybrid detector works correctly after retraining the stylometric RFs
and integrating the SOTA LLaMA 3.3 70B cloud-based classifier.
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from tool.hybrid_detector import predict_hybrid

def main():
    print("Testing SOTA Hybrid Detector...")
    
    # 1. Test with a typically AI-looking text
    ai_text = (
        "In today's fast-paced world, it is crucial to remember the vital importance of education. "
        "Furthermore, the results demonstrate that implementing this approach is clearly effective, "
        "allowing individuals to maximize their full potential. In conclusion, it is obvious that "
        "we must establish strong foundations for future success."
    )
    print("\nText 1 (Typically AI):")
    print(f"[{ai_text}]")
    try:
        prob = predict_hybrid(ai_text, register="news")
        print(f"=> AI Probability: {prob:.4f}")
    except Exception as e:
        print(f"=> FAILED: {e}")
        
    # 2. Test with a typically human-looking text
    human_text = (
        "i went to the store today and forgot to buy bread. my dog was waiting outside "
        "in the car and he was barking like crazy because he wanted to get home and eat his dinner."
    )
    print("\nText 2 (Typically Human):")
    print(f"[{human_text}]")
    try:
        prob = predict_hybrid(human_text, register="social")
        print(f"=> AI Probability: {prob:.4f}")
    except Exception as e:
        print(f"=> FAILED: {e}")

if __name__ == '__main__':
    main()
