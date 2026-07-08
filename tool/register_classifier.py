#!/usr/bin/env python3
"""Shared register classification utilities used by API and training scripts."""
import os
from typing import Dict, Any, Tuple, Optional


def load_models_from_manifest(models_dir: str):
    """Load register classifier and detectors from models/manifest.json."""
    import json
    import joblib
    manifest_path = os.path.join(models_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    models = {'detectors': {}}
    rc_path = os.path.join(models_dir, manifest['register_classifier'])
    if os.path.exists(rc_path):
        _rc = joblib.load(rc_path)
        models['register_classifier'] = _rc['model'] if isinstance(_rc, dict) else _rc
        if isinstance(_rc, dict) and 'label_encoder' in _rc:
            models['register_label_encoder'] = _rc['label_encoder']

    for register, filename in manifest['detectors'].items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            _d = joblib.load(path)
            models['detectors'][register] = _d['model'] if isinstance(_d, dict) else _d

    all_path = os.path.join(models_dir, manifest['all_register_detector'])
    if os.path.exists(all_path):
        _all = joblib.load(all_path)
        models['all_detector'] = _all['model'] if isinstance(_all, dict) else _all

    models['feature_cols'] = manifest.get('feature_cols')
    return models


def classify_register(feature_vector: list, models: Dict[str, Any], override: Optional[str] = None) -> Tuple[str, Optional[float]]:
    """
    Classify register for a feature vector.
    Returns (register_name, confidence).
    """
    import numpy as np
    if override:
        if override in models.get('detectors', {}):
            return override, None
        return override, None

    if 'register_classifier' not in models:
        return 'all', None

    X = np.array([feature_vector])
    reg_pred = models['register_classifier'].predict(X)[0]
    reg_proba = models['register_classifier'].predict_proba(X)[0]
    classes = models['register_classifier'].classes_
    idx = list(classes).index(reg_pred)
    confidence = float(reg_proba[idx])

    if 'register_label_encoder' in models:
        register = str(models['register_label_encoder'].inverse_transform([reg_pred])[0])
    else:
        register = str(reg_pred)
    return register, confidence
