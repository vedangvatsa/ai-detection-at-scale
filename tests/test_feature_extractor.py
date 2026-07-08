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


def test_use_pos_tags_default_is_backward_compatible():
    text = "The quick brown fox jumps over the lazy dog."
    feats_default = extract_features(text, extended=True)
    feats_legacy = extract_features(text, extended=True, use_pos_tags=False)
    assert set(feats_default.keys()) == set(feats_legacy.keys())
    # When use_pos_tags is not enabled, behavior must match the legacy suffix counts.
    assert feats_default['adjective_density'] == feats_legacy['adjective_density']
    assert feats_default['adverb_density'] == feats_legacy['adverb_density']
    assert feats_default['nominalization_density'] == feats_legacy['nominalization_density']


def test_use_pos_tags_true_does_not_crash():
    text = "The quick brown fox jumps over the lazy dog."
    feats = extract_features(text, extended=True, use_pos_tags=True)
    assert feats is not None
    # If nltk is not installed, the result should still contain the expected keys.
    assert 'adjective_density' in feats
    assert 'adverb_density' in feats
    assert 'nominalization_density' in feats


if __name__ == '__main__':
    test_feature_count_matches_code()
    test_negative_words_no_duplicates()
    test_capitalized_entity_ignores_lone_sentence_starter()
    test_extract_features_short_text_returns_none()
    test_use_pos_tags_default_is_backward_compatible()
    test_use_pos_tags_true_does_not_crash()
    print('All feature extractor tests passed.')
