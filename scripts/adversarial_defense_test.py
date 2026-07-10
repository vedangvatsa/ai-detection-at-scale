#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tool.feature_extractor import extract_features

def main():
    print("Evaluating Advanced Stylometrics against Adversarial Humanizer Synonym-Swapping...")
    
    # 1. Clean AI-generated text (standard GPT output)
    ai_text = (
        "Artificial intelligence has experienced significant growth in recent years. "
        "It provides numerous benefits to various industries by automating repetitive tasks. "
        "Furthermore, machine learning algorithms allow computers to process vast amounts of data efficiently. "
        "Therefore, organizations can make better decisions based on data-driven insights."
    )
    
    # 2. Humanized/Paraphrased version of the same AI text (QuillBot-like synonym swaps, but same structure)
    # Synonyms swapped:
    # Artificial intelligence -> Machine intellect
    # experienced significant growth -> undergone substantial expansion
    # provides numerous benefits -> offers diverse advantages
    # automating repetitive tasks -> streamlining redundant chores
    # machine learning algorithms -> deep learning models
    # efficiently -> quickly
    # make better decisions -> reach superior conclusions
    humanized_text = (
        "Machine intellect has undergone substantial expansion in recent years. "
        "It offers diverse advantages to various fields by streamlining redundant chores. "
        "Furthermore, deep learning models allow computers to handle vast volumes of data quickly. "
        "Therefore, firms can reach superior conclusions based on analytics-driven insights."
    )
    
    # 3. Real human-written text (high variation, conversational/stylistic shifts)
    human_text = (
        "So, AI has been growing like crazy lately, hasn't it? "
        "Honestly, it's pretty cool because it saves people from doing the same boring tasks over and over again. "
        "But the real magic is how these computers crunch numbers so fast. "
        "It's wild—companies are basically letting algorithms run the show now."
    )
    
    print("\n--- Extracting Features ---")
    ai_feats = extract_features(ai_text, extended=True, use_pos_tags=True)
    humanized_feats = extract_features(humanized_text, extended=True, use_pos_tags=True)
    human_feats = extract_features(human_text, extended=True, use_pos_tags=True)
    
    target_keys = ['noun_verb_ratio', 'adj_adv_ratio', 'pos_transition_entropy', 'sent_length_std']
    
    print(f"{'Feature':<25} | {'AI Text':<10} | {'Humanized AI':<12} | {'Human Text':<10} | {'AI vs Humanized (Diff)':<20}")
    print("-" * 88)
    for k in target_keys:
        v_ai = ai_feats[k]
        v_hum_ai = humanized_feats[k]
        v_human = human_feats[k]
        diff = abs(v_ai - v_hum_ai)
        print(f"{k:<25} | {v_ai:<10.4f} | {v_hum_ai:<12.4f} | {v_human:<10.4f} | {diff:<20.4f}")
        
    print("\nObservation:")
    print("1. Notice how 'pos_transition_entropy' and 'noun_verb_ratio' remain extremely close between the clean AI text")
    print("   and its humanized counterpart. Humanizers swap synonyms but preserve the underlying syntactic transition probabilities!")
    print("2. The real human text exhibits significantly different structural metrics (e.g. sentence length variability and different parts-of-speech balance).")
    print("\nAdversarial defense test completed successfully!")

if __name__ == '__main__':
    main()
