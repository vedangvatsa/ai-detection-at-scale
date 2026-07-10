#!/usr/bin/env python3
"""
Span-level AI text detection.

Detects which specific sentences or paragraphs in a document are AI-generated,
rather than classifying the entire document. This enables detection of AI-assisted
writing where humans and AI collaborated.

Algorithm:
  - Split text into sentences
  - Score each sentence via a sliding 3-sentence window (provides context)
  - Detect abrupt style-shift boundary transitions (human → AI handoffs)
  - Return per-sentence heat scores and transition points

Usage:
    from tool.span_detector import detect_spans
    result = detect_spans(text, detector, feature_cols=feature_cols)
"""
import re
import numpy as np
from typing import List, Optional


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences preserving structure."""
    # Handle common sentence boundaries while preserving abbreviations
    text = re.sub(r'\n{2,}', ' <PARA> ', text)
    text = re.sub(r'\n', ' ', text)
    # Split on sentence-ending punctuation
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = []
    for p in parts:
        sub_parts = p.split(' <PARA> ')
        for s in sub_parts:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def _detect_transitions(scores: List[Optional[float]], threshold: float = 0.25) -> List[dict]:
    """
    Detect abrupt style-shift transitions between consecutive sentences.
    A transition is flagged where the score jumps by more than `threshold`.
    """
    transitions = []
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    for idx in range(1, len(valid)):
        prev_i, prev_score = valid[idx - 1]
        curr_i, curr_score = valid[idx]
        delta = curr_score - prev_score
        if abs(delta) >= threshold:
            direction = "human_to_ai" if delta > 0 else "ai_to_human"
            transitions.append({
                "at_sentence_index": curr_i,
                "delta": round(delta, 3),
                "direction": direction,
                "from_probability": round(prev_score, 3),
                "to_probability": round(curr_score, 3),
            })
    return transitions


def detect_spans(
    text: str,
    detector,
    feature_cols: List[str],
    window_size: int = 3,
    min_words_per_window: int = 15,
) -> dict:
    """
    Detect AI-generated spans in a text.

    Args:
        text: Input document text.
        detector: Trained sklearn classifier with predict_proba.
        feature_cols: Feature column names expected by the detector.
        window_size: Number of sentences per sliding window (default 3).
        min_words_per_window: Minimum word count to attempt feature extraction.

    Returns:
        dict with keys:
            - sentences: list of {text, word_count, ai_probability, is_ai}
            - transitions: list of detected style-shift boundaries
            - ai_sentence_count: number of sentences flagged as AI
            - human_sentence_count: number of sentences flagged as human
            - mixed_document: bool — True if both AI and human spans detected
            - overall_ai_probability: document-level aggregate score
    """
    from tool.feature_extractor import extract_feature_vector

    sentences = _split_sentences(text)
    if not sentences:
        return {"sentences": [], "transitions": [], "mixed_document": False}

    scores = []
    for i, _ in enumerate(sentences):
        # Build context window
        start = max(0, i - (window_size // 2))
        end = min(len(sentences), i + (window_size // 2) + 1)
        window_text = " ".join(sentences[start:end])
        word_count = len(window_text.split())

        if word_count < min_words_per_window:
            scores.append(None)
            continue

        feat_vec = extract_feature_vector(window_text, feature_cols=feature_cols, extended=False)
        if feat_vec is None:
            scores.append(None)
            continue

        try:
            X = np.array([feat_vec])
            proba = detector.predict_proba(X)[0]
            classes = list(detector.classes_)
            ai_idx = classes.index(1) if 1 in classes else 1
            scores.append(float(proba[ai_idx]))
        except Exception:
            scores.append(None)

    # Fill None scores by interpolation from neighbors
    filled_scores = _fill_none_scores(scores)

    transitions = _detect_transitions(filled_scores, threshold=0.25)

    # Build per-sentence output
    sentence_results = []
    for i, (sent, score) in enumerate(zip(sentences, filled_scores)):
        word_count = len(sent.split())
        sentence_results.append({
            "index": i,
            "text": sent,
            "word_count": word_count,
            "ai_probability": round(score, 3) if score is not None else None,
            "is_ai": score >= 0.5 if score is not None else None,
            "uncertain": score is not None and 0.35 < score < 0.65,
        })

    valid_scores = [s for s in filled_scores if s is not None]
    overall = float(np.mean(valid_scores)) if valid_scores else 0.5

    ai_count = sum(1 for s in sentence_results if s["is_ai"] is True)
    human_count = sum(1 for s in sentence_results if s["is_ai"] is False)
    mixed = ai_count > 0 and human_count > 0

    return {
        "sentences": sentence_results,
        "transitions": transitions,
        "ai_sentence_count": ai_count,
        "human_sentence_count": human_count,
        "mixed_document": mixed,
        "overall_ai_probability": round(overall, 4),
    }


def _fill_none_scores(scores: List[Optional[float]]) -> List[Optional[float]]:
    """Fill None scores by linear interpolation from valid neighbors."""
    result = list(scores)
    n = len(result)
    for i in range(n):
        if result[i] is not None:
            continue
        # Find left and right valid neighbors
        left_val, left_i = None, None
        right_val, right_i = None, None
        for j in range(i - 1, -1, -1):
            if result[j] is not None:
                left_val, left_i = result[j], j
                break
        for j in range(i + 1, n):
            if result[j] is not None:
                right_val, right_i = result[j], j
                break

        if left_val is not None and right_val is not None:
            # Linear interpolation
            frac = (i - left_i) / (right_i - left_i)
            result[i] = left_val + frac * (right_val - left_val)
        elif left_val is not None:
            result[i] = left_val
        elif right_val is not None:
            result[i] = right_val
        # else stays None

    return result
