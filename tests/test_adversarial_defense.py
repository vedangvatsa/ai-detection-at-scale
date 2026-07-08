#!/usr/bin/env python3
"""Unit tests for adversarial defense normalization."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tool.adversarial_defense import normalize_text_defensive


def test_homoglyph_normalization():
    # Cyrillic 'а' (U+0430) looks like Latin 'a'
    text = "Thіs іs а tеst"  # mixed Cyrillic
    cleaned = normalize_text_defensive(text)
    assert cleaned == "This is a test", f"Got: {cleaned}"


def test_zero_width_chars_removed():
    text = "test \u200b\u200c text \ufeff"
    cleaned = normalize_text_defensive(text)
    assert cleaned == "test text", f"Got: {cleaned}"


def test_whitespace_attack_collapsed():
    text = "word1   \t\n\n   word2"
    cleaned = normalize_text_defensive(text)
    assert cleaned == "word1 word2", f"Got: {cleaned}"


def test_punctuation_noise_deduplicated():
    text = "Hello!!!???..."
    cleaned = normalize_text_defensive(text)
    assert cleaned == "Hello!?.", f"Got: {cleaned}"


def test_non_string_input():
    assert normalize_text_defensive(None) == ""


if __name__ == '__main__':
    test_homoglyph_normalization()
    test_zero_width_chars_removed()
    test_whitespace_attack_collapsed()
    test_punctuation_noise_deduplicated()
    test_non_string_input()
    print("All adversarial defense tests passed.")
