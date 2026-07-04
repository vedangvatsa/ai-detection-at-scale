#!/usr/bin/env python3
"""
Calibrate detection thresholds to hit target FPR with maximal TPR.
For each register, find the threshold that achieves exactly 0.1% FPR.
"""
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, normalize_unicode, ORIGINAL_FEATURE_COLS

TARGET_FPR = 0.001  # 0.1% FPR

def main():
    print("Loading feature corpus...")
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path, columns=['label', 'register'] + ORIGINAL_FEATURE_COLS)
    df = df.dropna(subset=ORIGINAL_FEATURE_COLS)
    print(f"Loaded {len(df)} texts")

    # Load models
    print("Loading models...")
    manifest_path = os.path.join(MODELS_DIR, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    detectors = {}
    for register, filename in manifest['detectors'].items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            _d = joblib.load(path)
            detectors[register] = _d['model'] if isinstance(_d, dict) else _d
            print(f"  Loaded {register} detector")

    # Calibrate per-register thresholds
    calibration_results = {}

    for register in ['academic', 'news', 'social', 'creative']:
        print(f"\n=== Calibrating {register} ===")
        df_reg = df[df['register'] == register]
        
        # Split by label
        human = df_reg[df_reg['label'] == 0]
        ai = df_reg[df_reg['label'] == 1]
        
        print(f"  Human: {len(human)}, AI: {len(ai)}")
        
        if register not in detectors:
            print(f"  WARNING: No detector for {register}")
            continue
        
        detector = detectors[register]
        
        # Get feature vectors
        X_human = human[ORIGINAL_FEATURE_COLS].values
        X_ai = ai[ORIGINAL_FEATURE_COLS].values
        
        # Get probabilities
        y_human = detector.predict_proba(X_human)[:, 1]  # AI probability
        y_ai = detector.predict_proba(X_ai)[:, 1]
        
        # Compute ROC curve on human samples (to find FPR threshold)
        # We want threshold where FPR = 0.001
        # FPR = P(pred_AI | human) = proportion of human samples with prob > threshold
        sorted_probs = np.sort(y_human)[::-1]  # Descending
        n_human = len(y_human)
        target_count = int(TARGET_FPR * n_human)
        
        if target_count == 0:
            threshold = 1.0  # Most conservative
        else:
            threshold = sorted_probs[target_count - 1]
        
        # Compute TPR at this threshold
        tpr = np.mean(y_ai > threshold)
        
        print(f"  Target FPR: {TARGET_FPR:.4f}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  TPR at threshold: {tpr:.4f}")
        
        calibration_results[register] = {
            'threshold': float(threshold),
            'target_fpr': TARGET_FPR,
            'tpr': float(tpr),
            'n_human': n_human,
            'n_ai': len(ai),
        }
    
    # Save calibration results
    calib_path = os.path.join(RESULTS_DIR, 'threshold_calibration.json')
    with open(calib_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    print(f"\nCalibration saved to {calib_path}")
    
    # Also save as CSV for easy viewing
    calib_df = pd.DataFrame(calibration_results).T
    calib_csv = os.path.join(RESULTS_DIR, 'threshold_calibration.csv')
    calib_df.to_csv(calib_csv)
    print(f"CSV saved to {calib_csv}")
    print(calib_df.to_string())

if __name__ == '__main__':
    main()
