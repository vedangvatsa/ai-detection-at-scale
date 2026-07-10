#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool.multilingual import extract_language_agnostic_features, detect_language

def main():
    print("Testing Language-Agnostic Feature Extraction...")
    
    # 1. Russian text
    ru_text = (
        "Искусственный интеллект переживает значительный рост в последние годы. "
        "Он предоставляет многочисленные преимущества различным отраслям за счет автоматизации рутинных задач. "
        "Кроме того, алгоритмы машинного обучения позволяют компьютерам эффективно обрабатывать большие объемы данных."
    )
    
    # 2. Japanese text
    ja_text = (
        "人工知能は近年、著しい成長を遂げています。 "
        "日常的なタスクを自動化することにより、さまざまな業界に多くのメリットをもたらします。 "
        "さらに、機械学習アルゴリズムにより、コンピュータは大量のデータを効率的に処理できます。"
    )
    
    for label, text in [("Russian", ru_text), ("Japanese", ja_text)]:
        lang = detect_language(text)
        print(f"\n{label} Text (Detected Lang: {lang}):")
        feats = extract_language_agnostic_features(text)
        if feats is None:
            print("  Extraction failed!")
            continue
        
        print(f"  Character Entropy: {feats['char_entropy']:.4f}")
        print(f"  Punctuation Entropy: {feats['punct_entropy']:.4f}")
        print(f"  Repetition Rate: {feats['rep_rate']:.4f}")
        print(f"  Mean Sentence Length: {feats['mean_sent_len']:.2f}")
        print(f"  Type-Token Ratio: {feats['type_token_ratio']:.4f}")
        
    print("\nLanguage-agnostic extraction tests completed successfully!")

if __name__ == '__main__':
    main()
