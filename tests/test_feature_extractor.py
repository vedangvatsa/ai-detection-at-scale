#!/usr/bin/env python3
"""Unit tests for the stylometric feature extractor."""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tool.feature_extractor import (
    extract_features,
    ORIGINAL_FEATURE_COLS,
    EXTENDED_FEATURE_COLS,
    ALL_FEATURE_COLS,
    NEGATIVE_WORDS,
)


def test_feature_count_matches_code():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Furthermore, the results demonstrate that this approach is clearly effective."
    )
    feats = extract_features(text, extended=True)
    assert feats is not None
    assert set(ALL_FEATURE_COLS) == set(feats.keys()), f"Missing: {set(ALL_FEATURE_COLS) - set(feats.keys())}"
    assert len(ALL_FEATURE_COLS) == 35, f"Expected 35 features, got {len(ALL_FEATURE_COLS)}"
    assert len(ORIGINAL_FEATURE_COLS) == 11
    assert len(EXTENDED_FEATURE_COLS) == 24


def test_negative_words_no_duplicates():
    assert len(NEGATIVE_WORDS) == len(set(NEGATIVE_WORDS)), "NEGATIVE_WORDS contains duplicates"


def test_capitalized_entity_ignores_lone_sentence_starter():
    # "The" at the start of a sentence should not count as an entity.
    text_lone = "The weather is nice today. It is sunny."
    feats_lone = extract_features(text_lone, extended=True)

    text_entity = "The United States is a country. It is large."
    feats_entity = extract_features(text_entity, extended=True)

    assert feats_lone is not None and feats_entity is not None
    assert feats_entity['capitalized_entity_density'] > feats_lone['capitalized_entity_density']


def test_extract_features_short_text_returns_none():
    assert extract_features("hi", extended=True) is None
    assert extract_features("", extended=True) is None


if __name__ == '__main__':
    test_feature_count_matches_code()
    test_negative_words_no_duplicates()
    test_capitalized_entity_ignores_lone_sentence_starter()
    test_extract_features_short_text_returns_none()
    print('All feature extractor tests passed.')
