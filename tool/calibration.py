import os
import math
import joblib
import numpy as np

_CALIBRATION_MODEL = None

def _load_calibration_model():
    global _CALIBRATION_MODEL
    if _CALIBRATION_MODEL is not None:
        return _CALIBRATION_MODEL
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'models', 'calibration_model.joblib'
    )
    if os.path.exists(model_path):
        try:
            _CALIBRATION_MODEL = joblib.load(model_path)
        except Exception:
            _CALIBRATION_MODEL = None
    return _CALIBRATION_MODEL


def _heuristic_calibrate(probability: float, word_count: int, target_length: int = 50, steepness: float = 0.08) -> float:
    """Original length-conditioned heuristic fallback."""
    if word_count >= target_length:
        return probability

    x = word_count - (target_length / 2.0)
    weight = 1.0 / (1.0 + math.exp(-steepness * x))
    calibrated = 0.5 + weight * (probability - 0.5)
    return float(calibrated)


def calibrate_probability(probability: float, word_count: int, target_length: int = 50, steepness: float = 0.08) -> float:
    """
    Calibrates prediction probability.

    If a trained calibration model exists (trained by scripts/train_calibration.py),
    it is used. The model can be Platt-scaled logistic regression or isotonic
    regression fit on held-out predictions. Otherwise, falls back to the original
    length-conditioned heuristic that pulls short-document probabilities toward 0.5.
    """
    model = _load_calibration_model()
    if model is None:
        return _heuristic_calibrate(probability, word_count, target_length, steepness)

    try:
        cal_type = model.get('type', 'platt')
        if cal_type == 'platt':
            clf = model['platt']
            X = np.array([[probability, word_count, probability * word_count]])
            calibrated = clf.predict_proba(X)[0, 1]
        else:
            iso = model['iso']
            calibrated = float(iso.predict([probability])[0])
        return float(np.clip(calibrated, 0.0, 1.0))
    except Exception:
        return _heuristic_calibrate(probability, word_count, target_length, steepness)
