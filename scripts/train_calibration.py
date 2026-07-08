#!/usr/bin/env python3
"""
Train a length-aware probability calibrator on held-out stylometric predictions.

This replaces the hand-tuned sigmoid heuristic in tool/calibration.py with a
model-based calibrator: a LogisticRegression (Platt scaling) conditioned on the
raw detector probability and the document word count. It also evaluates
calibration with Brier score and Expected Calibration Error (ECE).
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, ORIGINAL_FEATURE_COLS
from tool.register_classifier import load_models_from_manifest, classify_register

FEATURE_COLS = ORIGINAL_FEATURE_COLS


def _ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop = in_bin.astype(float).mean()
        if prop > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop
    return float(ece)


def _word_count(text):
    return len(str(text).split()) if isinstance(text, str) and text.strip() else 0


def load_predictions_and_labels(df, models, feature_cols):
    """Return raw probabilities and labels using the loaded register detectors."""
    probs = []
    labels = []
    wcs = []
    detectors = models.get('detectors', {})
    all_detector = models.get('all_detector')

    for _, row in df.iterrows():
        text = str(row.get('text', ''))
        label = int(row.get('label', row.get('is_ai', 0)))
        feats = extract_feature_vector(text, feature_cols=feature_cols, extended=False)
        if feats is None:
            continue
        register, _ = classify_register(feats, models)
        reg_detector = detectors.get(register, all_detector)
        if reg_detector is None:
            continue
        proba = reg_detector.predict_proba([feats])[0]
        classes = reg_detector.classes_
        ai_label_idx = list(classes).index(1) if 1 in classes else 1
        raw_prob = float(proba[ai_label_idx])
        probs.append(raw_prob)
        labels.append(label)
        wcs.append(_word_count(text))
    return np.array(probs), np.array(labels), np.array(wcs)


def main():
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} not found.")
        return

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading models...")
    try:
        models = load_models_from_manifest(MODELS_DIR)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return
    feature_cols = models.get('feature_cols') or FEATURE_COLS

    print("Loading features...")
    df = pd.read_parquet(feat_path)
    required = ['text', 'label'] + feature_cols
    if 'label' not in df.columns and 'is_ai' in df.columns:
        df['label'] = df['is_ai']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}; falling back to feature-only calibration.")
        df = df.dropna(subset=feature_cols)
        X = df[feature_cols].values
        y = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        # Proxy for word count; use mean_sent_len as a noisy correlate.
        wcs = df['mean_sent_len'].fillna(20).values.astype(int)
        # Build a simple detector so we can emit probabilities
        from sklearn.ensemble import RandomForestClassifier
        det = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        X_tr, X_te, y_tr, y_te, wcs_tr, wcs_te = train_test_split(
            X, y, wcs, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        det.fit(X_tr, y_tr)
        raw_probs_tr = det.predict_proba(X_tr)[:, 1]
        raw_probs_te = det.predict_proba(X_te)[:, 1]
    else:
        df = df.dropna(subset=required)
        print(f"Loaded {len(df)} samples.")
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
        if not models.get('detectors') and not models.get('all_detector'):
            print("No register detectors found; cannot train calibrator.")
            return
        raw_probs_tr, y_tr, wcs_tr = load_predictions_and_labels(train_df, models, feature_cols)
        raw_probs_te, y_te, wcs_te = load_predictions_and_labels(test_df, models, feature_cols)

    print(f"Calibration samples: train={len(raw_probs_tr)}, test={len(raw_probs_te)}")

    # Build calibration features: raw prob + word count (and interaction)
    X_cal_tr = np.column_stack([raw_probs_tr, wcs_tr, raw_probs_tr * wcs_tr])
    X_cal_te = np.column_stack([raw_probs_te, wcs_te, raw_probs_te * wcs_te])

    # Train Platt-style calibrator
    platt = LogisticRegression(max_iter=1000, random_state=42)
    platt.fit(X_cal_tr, y_tr)

    # Train isotonic calibrator for comparison
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs_tr, y_tr)

    calibrated_platt = platt.predict_proba(X_cal_te)[:, 1]
    calibrated_iso = iso.predict(raw_probs_te)

    # Evaluate
    metrics = {
        'raw_brier': float(brier_score_loss(y_te, raw_probs_te)),
        'platt_brier': float(brier_score_loss(y_te, calibrated_platt)),
        'iso_brier': float(brier_score_loss(y_te, calibrated_iso)),
        'raw_ece': _ece(y_te, raw_probs_te),
        'platt_ece': _ece(y_te, calibrated_platt),
        'iso_ece': _ece(y_te, calibrated_iso),
        'raw_log_loss': float(log_loss(y_te, np.clip(raw_probs_te, 1e-6, 1 - 1e-6))),
        'platt_log_loss': float(log_loss(y_te, np.clip(calibrated_platt, 1e-6, 1 - 1e-6))),
    }
    print("\nCalibration metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.5f}")

    # Save the best calibrator (choose Platt by default; lower Brier is better)
    best_name = 'platt' if metrics['platt_brier'] <= metrics['iso_brier'] else 'iso'
    print(f"\nBest calibrator: {best_name}")
    calibrator = {
        'platt': platt,
        'iso': iso,
        'type': best_name,
        'metrics': metrics,
    }
    out_path = os.path.join(MODELS_DIR, 'calibration_model.joblib')
    joblib.dump(calibrator, out_path)
    print(f"Saved calibrator to {out_path}")

    metrics_path = os.path.join(RESULTS_DIR, 'calibration_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
