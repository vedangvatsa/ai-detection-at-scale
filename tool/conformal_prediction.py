#!/usr/bin/env python3
"""
Conformal prediction confidence intervals for AI text detection.

Provides statistically valid prediction sets/intervals using inductive conformal
prediction (split conformal). Unlike heuristic calibration, conformal prediction
gives a formal coverage guarantee:

    P(true_label ∈ prediction_set) >= 1 - alpha

Usage:
    # At startup: fit on a held-out calibration set
    cp = ConformalPredictor()
    cp.fit(cal_probs, cal_labels)  # cal_probs: [0..1], cal_labels: [0 or 1]
    cp.save("models/conformal_predictor.joblib")

    # At inference: get interval
    lower, upper = cp.predict_interval(ai_probability, alpha=0.1)

Reference:
    Venn-Abers predictor / split conformal prediction.
    Angelopoulos & Bates (2021). "A gentle introduction to conformal prediction."
    https://arxiv.org/abs/2107.07511
"""
import os
import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """
    Split conformal predictor for binary AI-detection scores.

    Uses the nonconformity score s_i = |y_i - p_i| on a held-out calibration
    set to compute empirical quantiles, then constructs prediction intervals:

        [p - q_{1-alpha}, p + q_{1-alpha}]  clipped to [0, 1]

    This gives marginal coverage guarantees under exchangeability.
    """

    def __init__(self):
        self._cal_scores: Optional[np.ndarray] = None
        self._n_cal: int = 0

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> "ConformalPredictor":
        """
        Fit on a calibration split.

        Args:
            cal_probs: Predicted AI probabilities on calibration examples.
            cal_labels: True binary labels (1 = AI, 0 = human).
        """
        cal_probs = np.asarray(cal_probs, dtype=float)
        cal_labels = np.asarray(cal_labels, dtype=float)
        # Nonconformity score: residual from the true class probability
        self._cal_scores = np.abs(cal_labels - cal_probs)
        self._n_cal = len(cal_probs)
        return self

    def _quantile(self, alpha: float) -> float:
        """
        Compute the (1 - alpha) * (1 + 1/n) empirical quantile of calibration scores.
        The +1/n correction ensures valid coverage.
        """
        if self._cal_scores is None or self._n_cal == 0:
            return 0.5  # fallback: wide interval
        n = self._n_cal
        level = min(1.0, (1 - alpha) * (1 + 1 / n))
        return float(np.quantile(self._cal_scores, level))

    def predict_interval(
        self, ai_probability: float, alpha: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute a (1 - alpha) confidence interval for the AI probability.

        Args:
            ai_probability: Point estimate from the detector.
            alpha: Desired miscoverage rate. 0.1 → 90% coverage interval.

        Returns:
            (lower, upper) clipped to [0, 1].
        """
        q = self._quantile(alpha)
        lower = max(0.0, ai_probability - q)
        upper = min(1.0, ai_probability + q)
        return round(lower, 4), round(upper, 4)

    def predict_set(self, ai_probability: float, alpha: float = 0.1) -> dict:
        """
        Return full prediction metadata.

        Returns:
            {
                lower: float,
                upper: float,
                width: float,
                coverage_guarantee: float,   # 1 - alpha
                n_calibration: int,
                is_uncertain: bool,          # True if interval straddles 0.5
            }
        """
        lower, upper = self.predict_interval(ai_probability, alpha)
        straddles = lower < 0.5 < upper
        return {
            "lower": lower,
            "upper": upper,
            "width": round(upper - lower, 4),
            "coverage_guarantee": round(1 - alpha, 2),
            "n_calibration": self._n_cal,
            "is_uncertain": straddles,
        }

    def save(self, path: str):
        import joblib
        joblib.dump({"cal_scores": self._cal_scores, "n_cal": self._n_cal}, path)
        print(f"Conformal predictor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalPredictor":
        import joblib
        data = joblib.load(path)
        obj = cls()
        obj._cal_scores = data["cal_scores"]
        obj._n_cal = data["n_cal"]
        return obj


# ── Module-level singleton ─────────────────────────────────────────────────

_CP: Optional[ConformalPredictor] = None
_CP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "conformal_predictor.joblib"
)


def _load_cp() -> Optional[ConformalPredictor]:
    global _CP
    if _CP is not None:
        return _CP
    if os.path.exists(_CP_PATH):
        try:
            _CP = ConformalPredictor.load(_CP_PATH)
            print(f"  Loaded conformal predictor (n_cal={_CP._n_cal})")
        except Exception as e:
            print(f"  Warning: could not load conformal predictor: {e}")
    return _CP


def get_confidence_interval(
    ai_probability: float, alpha: float = 0.1
) -> Optional[dict]:
    """
    Get a confidence interval for an AI probability estimate.

    Returns None if the conformal predictor has not been fitted and saved yet.
    Falls back to a heuristic symmetric interval based on distance from 0.5.
    """
    cp = _load_cp()
    if cp is not None:
        return cp.predict_set(ai_probability, alpha=alpha)

    # Heuristic fallback: wider interval near 0.5 (high uncertainty), narrower at extremes
    uncertainty = 1 - 2 * abs(ai_probability - 0.5)  # 0 at extremes, 1 at 0.5
    half_width = 0.05 + 0.20 * uncertainty
    lower = max(0.0, round(ai_probability - half_width, 4))
    upper = min(1.0, round(ai_probability + half_width, 4))
    return {
        "lower": lower,
        "upper": upper,
        "width": round(upper - lower, 4),
        "coverage_guarantee": None,
        "n_calibration": 0,
        "is_uncertain": lower < 0.5 < upper,
        "method": "heuristic_fallback",
    }
