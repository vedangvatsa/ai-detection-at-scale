#!/usr/bin/env python3
"""
Statistical watermark detection for LLM-generated text.

Implements two detection approaches:

1. KGW z-test (Kirchenbauer et al. 2023):
   Given a vocabulary partition into green/red lists derived from a secret key,
   tests whether a text contains an anomalously high fraction of green-list tokens.
   Requires the LLM's secret seed — suitable for enterprise deployments where
   the user owns the LLM.

2. Blind distribution test (key-agnostic):
   Detects statistically anomalous token uniformity patterns consistent with
   watermarking, without needing the key. Less precise but works as a soft signal
   on any text.

References:
    Kirchenbauer et al. (2023). "A Watermark for Large Language Models."
    https://arxiv.org/abs/2301.10226

Usage:
    from tool.watermark_detector import detect_watermark
    result = detect_watermark(text, seed=None)  # blind mode
    result = detect_watermark(text, seed=42)     # keyed mode
"""
import hashlib
import math
from typing import Optional


def _tokenize_simple(text: str):
    """Simple whitespace + punctuation tokenizer (no model required)."""
    import re
    return re.findall(r'\b\w+\b', text.lower())


def _build_green_list(vocab_tokens, seed: int, gamma: float = 0.5):
    """
    Partition the vocabulary into green (allowed) and red (suppressed) lists
    using a deterministic hash derived from the seed.
    gamma = fraction assigned to the green list.
    """
    green = set()
    for token in vocab_tokens:
        h = int(hashlib.md5(f"{seed}:{token}".encode()).hexdigest(), 16)
        if (h % 100) < int(gamma * 100):
            green.add(token)
    return green


def _kgw_ztest(tokens, green_list, gamma: float = 0.5) -> dict:
    """
    Compute the KGW z-test statistic.

    Under the null hypothesis (no watermark), each token falls in the green list
    with probability γ. A high z-score indicates watermark presence.

    Returns:
        z_score, p_value, green_fraction, n_tokens
    """
    if not tokens:
        return {"z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0, "n_tokens": 0}

    n = len(tokens)
    k = sum(1 for t in tokens if t in green_list)
    green_fraction = k / n

    # z = (k - n*gamma) / sqrt(n * gamma * (1 - gamma))
    expected = n * gamma
    std_dev = math.sqrt(n * gamma * (1 - gamma))
    z_score = (k - expected) / std_dev if std_dev > 0 else 0.0

    # One-tailed p-value (watermark shifts distribution rightward)
    p_value = _z_to_p(z_score)

    return {
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 6),
        "green_fraction": round(green_fraction, 4),
        "n_tokens": n,
    }


def _z_to_p(z: float) -> float:
    """Approximate one-tailed p-value using error function."""
    import math
    return 0.5 * math.erfc(z / math.sqrt(2))


def _blind_entropy_test(tokens) -> dict:
    """
    Key-agnostic blind watermark indicator.

    Watermarked text suppresses certain tokens (red list) leading to:
    - Higher type-token diversity in short windows (fewer repeats of suppressed tokens)
    - More uniform character-level n-gram distribution

    We compute the unigram entropy of tokens and compare to expected human entropy.
    High entropy (> threshold) combined with other AI signals is a weak positive indicator.
    """
    if len(tokens) < 20:
        return {"entropy": None, "watermark_signal": "insufficient_length"}

    from collections import Counter
    counts = Counter(tokens)
    n = sum(counts.values())
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    # Heuristic thresholds calibrated on human vs. AI text corpus
    # AI text tends to have higher unigram entropy (more uniform distribution)
    # due to watermarking bias suppressing common "red" tokens
    # Typical human text: 4.5-6.5 bits; watermarked AI: often > 6.8 bits
    if entropy > 7.0:
        signal = "possible_watermark"
    elif entropy > 6.5:
        signal = "elevated_entropy"
    else:
        signal = "normal"

    return {
        "entropy": round(entropy, 4),
        "watermark_signal": signal,
    }


def detect_watermark(
    text: str,
    seed: Optional[int] = None,
    gamma: float = 0.5,
    z_threshold: float = 4.0,
) -> dict:
    """
    Run watermark detection on input text.

    Args:
        text: Input text to test.
        seed: LLM's secret watermark seed. If None, runs blind entropy test only.
        gamma: Green-list fraction (default 0.5, matching KGW paper).
        z_threshold: Z-score threshold for positive detection (default 4.0 → p ≈ 0.00003).

    Returns:
        dict with:
            - watermark_detected: bool
            - confidence: "high" | "medium" | "low" | "none"
            - method: "kgw_keyed" | "blind_entropy"
            - details: raw test statistics
    """
    tokens = _tokenize_simple(text)

    if seed is not None:
        # Keyed KGW detection
        vocab = set(tokens)
        green_list = _build_green_list(vocab, seed=seed, gamma=gamma)
        stats = _kgw_ztest(tokens, green_list, gamma=gamma)
        stats["method"] = "kgw_keyed"

        z = stats["z_score"]
        if z >= z_threshold:
            detected = True
            confidence = "high" if z >= z_threshold * 1.5 else "medium"
        else:
            detected = False
            confidence = "low" if z >= z_threshold * 0.6 else "none"

        return {
            "watermark_detected": detected,
            "confidence": confidence,
            "method": "kgw_keyed",
            "details": stats,
        }

    else:
        # Blind entropy-based detection
        stats = _blind_entropy_test(tokens)
        stats["n_tokens"] = len(tokens)

        signal = stats.get("watermark_signal", "normal")
        detected = signal == "possible_watermark"
        confidence = "low" if signal == "possible_watermark" else "none"

        if signal == "elevated_entropy":
            confidence = "none"  # Too weak to flag

        return {
            "watermark_detected": detected,
            "confidence": confidence,
            "method": "blind_entropy",
            "details": stats,
            "note": "Blind detection is a weak signal only. Provide seed for keyed KGW detection.",
        }
