#!/usr/bin/env python3
"""
Sentence-level AI detection analyzer.
Uses overlapping sliding windows to detect localized AI text portions
and provides character-level offsets for frontend rendering.
"""
import re
import numpy as np
from tool.feature_extractor import extract_feature_vector, ORIGINAL_FEATURE_COLS

def split_sentences_with_offsets(text: str):
    """
    Splits text into sentences while preserving exact character start/end offsets.
    Returns a list of dicts: [{"text": str, "start": int, "end": int}]
    """
    if not text:
        return []
        
    # Pattern to match sentence endings followed by optional quotes/spaces or end of text
    pattern = re.compile(r'[^.!?]*[.!?]+["\'\s]*|[^.!?]+$')
    sentences = []
    
    for match in pattern.finditer(text):
        sent_text = match.group(0)
        # Skip empty or whitespace-only sentences
        if not sent_text.strip() or not re.search(r'\w', sent_text):
            continue
        sentences.append({
            "text": sent_text,
            "start": match.start(),
            "end": match.end()
        })
        
    # Fallback if no sentences matched but there is text
    if not sentences and text.strip():
        sentences.append({
            "text": text,
            "start": 0,
            "end": len(text)
        })
        
    return sentences

def analyze_sentences(text: str, detector, feature_cols=None):
    """
    Analyzes sentences in a document using a sliding window.
    Assigns an AI probability to each sentence.
    """
    if feature_cols is None:
        feature_cols = ORIGINAL_FEATURE_COLS
        
    sentences = split_sentences_with_offsets(text)
    if not sentences:
        return []
        
    n_sents = len(sentences)
    scores = np.zeros(n_sents)
    counts = np.zeros(n_sents) # Track how many windows covered this sentence
    
    # We use a sliding window of size 3 (contextual sentences)
    window_size = 3
    
    for i in range(n_sents):
        # Determine sliding window bounds
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_sents, start_idx + window_size)
        # Adjust start if we hit the end bound
        if end_idx - start_idx < window_size:
            start_idx = max(0, end_idx - window_size)
            
        # Re-concatenate window sentences
        window_sents = sentences[start_idx:end_idx]
        window_text = "".join([s["text"] for s in window_sents])
        
        # Extract features
        feats = extract_feature_vector(window_text, feature_cols=feature_cols, extended=False)
        
        if feats is not None:
            X = np.array([feats])
            prob = float(detector.predict_proba(X)[0][1])
        else:
            # Fallback if text is too short: use a default or overall indicator
            # Let's try individual sentence extraction first
            individual_feats = extract_feature_vector(sentences[i]["text"], feature_cols=feature_cols, extended=False)
            if individual_feats is not None:
                X = np.array([individual_feats])
                prob = float(detector.predict_proba(X)[0][1])
            else:
                prob = 0.5 # Neutral
                
        # Accumulate score for all sentences in the current window
        for j in range(start_idx, end_idx):
            scores[j] += prob
            counts[j] += 1
            
    # Compute average scores
    results = []
    for i, sent in enumerate(sentences):
        avg_prob = float(scores[i] / counts[i]) if counts[i] > 0 else 0.5
        results.append({
            "text": sent["text"],
            "start": sent["start"],
            "end": sent["end"],
            "ai_probability": round(avg_prob, 4)
        })
        
    return results

if __name__ == "__main__":
    # Test sentence splitting
    test_text = "This is a sentence. And here is another! Is this a third sentence? Yes."
    sents = split_sentences_with_offsets(test_text)
    print("Split result:")
    for s in sents:
        print(f"[{s['start']}:{s['end']}] -> {repr(s['text'])}")
