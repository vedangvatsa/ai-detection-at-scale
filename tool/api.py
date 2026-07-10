#!/usr/bin/env python3
"""
Inference API for AI text detection.

Wraps pre-trained joblib models in a FastAPI REST endpoint.
Pipeline: Unicode normalization → feature extraction → register classification → per-register detection → probability score.

Usage:
    uvicorn tool.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /detect          — detect single text
    POST /detect/batch    — detect multiple texts
    GET  /health          — health check
    GET  /models          — list loaded models
    GET  /features        — list feature names
"""
import os
import time
import json
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tool.api_security import (
    get_cache_key,
    get_cache,
    set_cache,
    _check_api_key,
    check_rate_limit,
)

from tool.feature_extractor import (
    extract_feature_vector,
    ORIGINAL_FEATURE_COLS,
    ALL_FEATURE_COLS,
)

from tool.neural_detector import compute_perplexity_and_burstiness
from tool.sentence_analyzer import analyze_sentences
from tool.attribution import attribute_source
from tool.adversarial_defense import normalize_text_defensive
from tool.calibration import calibrate_probability
from tool.register_classifier import classify_register

SENSITIVITY_THRESHOLDS = {'high': 0.35, 'normal': 0.50, 'low': 0.70}

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# ── Model loading ──────────────────────────────────────────────────────────

_feature_norms = None

def load_feature_norms():
    global _feature_norms
    if _feature_norms is not None:
        return _feature_norms
    
    norms_path = os.path.join(PROJECT_DIR, 'results', 'effect_sizes.csv')
    if not os.path.exists(norms_path):
        _feature_norms = {}
        return _feature_norms
        
    norms = {}
    import csv
    with open(norms_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            reg = row['register']
            feat = row['feature']
            if reg not in norms:
                norms[reg] = {}
            try:
                norms[reg][feat] = {
                    'human_mean': float(row['human_mean']),
                    'human_sd': float(row['human_sd']),
                    'ai_mean': float(row['ai_mean']),
                    'ai_sd': float(row['ai_sd']),
                    'cohens_d': float(row['cohens_d'])
                }
            except ValueError:
                continue
    _feature_norms = norms
    return _feature_norms

def load_models():
    """Load all pre-trained models from the models/ directory."""
    manifest_path = os.path.join(MODELS_DIR, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    models = {}

    # Register classifier
    rc_path = os.path.join(MODELS_DIR, manifest['register_classifier'])
    if os.path.exists(rc_path):
        _rc = joblib.load(rc_path)
        models['register_classifier'] = _rc['model'] if isinstance(_rc, dict) else _rc
        if isinstance(_rc, dict) and 'label_encoder' in _rc:
            models['register_label_encoder'] = _rc['label_encoder']
        print(f"  Loaded register classifier ({os.path.getsize(rc_path) / 1e9:.2f} GB)")
    else:
        print(f"  WARNING: register classifier not found at {rc_path}")

    # Per-register detectors
    models['detectors'] = {}
    for register, filename in manifest['detectors'].items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            _d = joblib.load(path)
            models['detectors'][register] = _d['model'] if isinstance(_d, dict) else _d
            print(f"  Loaded {register} detector ({os.path.getsize(path) / 1e6:.0f} MB)")
        else:
            print(f"  WARNING: {register} detector not found at {path}")

    # All-register fallback detector
    all_path = os.path.join(MODELS_DIR, manifest['all_register_detector'])
    if os.path.exists(all_path):
        _all = joblib.load(all_path)
        models['all_detector'] = _all['model'] if isinstance(_all, dict) else _all
        print(f"  Loaded all-register detector ({os.path.getsize(all_path) / 1e9:.2f} GB)")
    else:
        print(f"  WARNING: all-register detector not found at {all_path}")

    # Check if SOTA hybrid ensemblers are available (either register ensembler + ONNX, or single ensembler + PyTorch semantic model)
    hybrid_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')
    onnx_path = os.path.join(MODELS_DIR, 'roberta_large_onnx_quantized.onnx')
    single_path = os.path.join(MODELS_DIR, 'beemo_hybrid_ensembler.joblib')
    pytorch_path = os.path.join(MODELS_DIR, 'roberta_large_semantic_model')
    
    if (os.path.exists(hybrid_path) and os.path.exists(onnx_path)) or (os.path.exists(single_path) and os.path.exists(pytorch_path)):
        models['hybrid_available'] = True
        print("  SOTA Hybrid Register-Aware detector is AVAILABLE and active.")
    else:
        models['hybrid_available'] = False

    # Homoglyph normalizer (if it's a callable)
    hg_path = os.path.join(MODELS_DIR, manifest.get('homoglyph_normalizer', ''))
    if os.path.exists(hg_path) and hg_path:
        try:
            models['homoglyph_normalizer'] = joblib.load(hg_path)
        except Exception:
            pass

    models['manifest'] = manifest
    models['feature_cols'] = manifest['feature_cols']
    models['registers'] = manifest['registers']

    return models


# ── Pydantic models ───────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")
    register: Optional[str] = Field(None, description="Override register (academic, news, social, creative). If None, auto-detected.")
    return_features: bool = Field(False, description="Return extracted feature values")
    sensitivity: str = Field("normal", description="Detection sensitivity level: low, normal, or high")


class DetectResponse(BaseModel):
    ai_probability: float = Field(..., description="Probability that text is AI-generated (0.0 to 1.0)")
    register: str = Field(..., description="Detected or specified register")
    register_confidence: Optional[float] = Field(None, description="Confidence of register classification")
    is_ai: bool = Field(..., description="Binary classification based on chosen sensitivity threshold")
    features: Optional[dict] = Field(None, description="Extracted feature values (if return_features=True)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    sentences: Optional[List[dict]] = Field(None, description="Sentence-level highlight heatmap")
    neural_signals: Optional[dict] = Field(None, description="Neural signals including perplexity and burstiness")
    model_attribution: Optional[dict] = Field(None, description="Predicted model source and confidence")
    is_calibrated: bool = Field(True, description="True if output probability is calibrated")
    sensitivity_applied: str = Field("normal", description="The sensitivity level used for the decision threshold")
    explainability: Optional[dict] = Field(None, description="Detailed stylometric feature explainability relative to human register norms")
    cascade_level: Optional[str] = Field(None, description="The level of the cascading model hierarchy used for prediction")


class BatchDetectRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    register: Optional[str] = Field(None, description="Override register for all texts")
    return_features: bool = Field(False, description="Return extracted feature values")
    sensitivity: str = Field("normal", description="Detection sensitivity level: low, normal, or high")


class BatchDetectResponse(BaseModel):
    results: List[DetectResponse]
    total_time_ms: float
    texts_processed: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    registers_available: List[str]


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Text Detection API",
    description="Stylometric AI text detection with register-aware ensemble routing.",
    version="1.0.0",
)

# CORS configuration (restrict in production via CORS_ORIGINS env var)
_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.join(SCRIPT_DIR, 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

_models = None


def get_models():
    global _models
    if _models is None:
        print("Loading models...")
        _models = load_models()
        print("Models loaded.")
    return _models


@app.on_event("startup")
def startup():
    get_models()


@app.get("/health", response_model=HealthResponse)
def health():
    m = get_models()
    return HealthResponse(
        status="ok",
        models_loaded=list(m.get('detectors', {}).keys()) + ['all_register', 'register_classifier'],
        registers_available=m.get('registers', []),
    )


@app.get("/models", response_model=HealthResponse)
def models_info():
    return health()


@app.get("/features")
def features():
    m = get_models()
    return {
        "original_features": ORIGINAL_FEATURE_COLS,
        "model_features": m['feature_cols'],
        "all_available_features": ALL_FEATURE_COLS,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest, request: Request):
    if not _check_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    # Rate Limiting
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Limit: 60 req/min.")
        
    # Caching
    cache_key = get_cache_key(req.text, req.register)
    cached = get_cache(cache_key)
    if cached is not None:
        cached["processing_time_ms"] = 0.0
        return cached

    m = get_models()
    t0 = time.time()

    # Step 1: Adversarial character normalization
    text = normalize_text_defensive(req.text)

    # Step 2: Feature extraction (using model's feature cols)
    feat_vector = extract_feature_vector(text, feature_cols=m['feature_cols'], extended=False)
    if feat_vector is None:
        raise HTTPException(status_code=400, detail="Text too short for feature extraction (minimum 20 words).")

    X = np.array([feat_vector])

    # Step 3: Register classification (or override)
    if req.register:
        register = req.register
        register_confidence = None
        if register not in m['detectors']:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown register '{register}'. Available: {list(m['detectors'].keys())}",
            )
    else:
        register, register_confidence = classify_register(feat_vector, m)

    # Step 4: Per-register detection / Hybrid ensembler
    if register in m['detectors']:
        detector = m['detectors'][register]
    elif 'all_detector' in m:
        detector = m['all_detector']
        register = 'all'
    else:
        detector = None

    # Step 4: Cascading Classifier Architecture
    if detector is None:
        raise HTTPException(status_code=500, detail="No detector available.")
        
    # Level 1: Fast Stylometrics
    fast_proba = detector.predict_proba(X)[0]
    classes = detector.classes_
    ai_label_idx = list(classes).index(1) if 1 in classes else 1
    fast_probability = float(fast_proba[ai_label_idx])
    
    ai_probability = None
    cascade_level = None
    
    # Fast-path check (high confidence threshold: <= 0.15 or >= 0.85)
    if fast_probability <= 0.15 or fast_probability >= 0.85 or not m.get('hybrid_available'):
        ai_probability = fast_probability
        cascade_level = "level_1_stylometrics"
    else:
        # Cascade to Level 2: Quantized ONNX & Hybrid Ensemble (heavy features)
        try:
            from tool.hybrid_detector import predict_hybrid
            ai_probability = predict_hybrid(text, register=register)
            cascade_level = "level_2_hybrid_ensemble"
        except Exception as e:
            print(f"Error executing hybrid prediction: {e}. Falling back to Level 1 stylometrics.")
            ai_probability = fast_probability
            cascade_level = "level_1_stylometrics"

    # Calibration: adjust based on word count to prevent short-text false positives
    word_count = len(text.split())
    ai_probability = calibrate_probability(ai_probability, word_count)

    # Step 5: Sentence-level analysis
    sentences_data = analyze_sentences(text, detector, feature_cols=m['feature_cols']) if detector is not None else []

    # Step 6: Neural perplexity & burstiness (safe fallback if model not cached)
    try:
        neural_signals = compute_perplexity_and_burstiness(text)
    except Exception as e:
        print(f"Neural signal computation failed: {e}", file=sys.stderr)
        neural_signals = {"perplexity": None, "burstiness": None, "error": str(e)}

    # Step 7: Model Attribution
    attribution = attribute_source(feat_vector, ai_probability)

    # Step 8: Build response
    elapsed_ms = (time.time() - t0) * 1000

    features_dict = None
    feats = extract_feature_vector(text, feature_cols=m['feature_cols'], extended=False)
    if feats:
        features_dict = dict(zip(m['feature_cols'], feats))

    # Compute Z-score explainability relative to human register norms
    explainability_data = {}
    norms = load_feature_norms()
    reg_norms = norms.get(register, norms.get('all', {}))
    
    if features_dict and reg_norms:
        for feat_name, val in features_dict.items():
            if feat_name in reg_norms:
                mean = reg_norms[feat_name]['human_mean']
                sd = reg_norms[feat_name]['human_sd']
                z_score = (val - mean) / sd if sd > 0 else 0.0
                cohens_d = reg_norms[feat_name]['cohens_d']
                
                # Check deviation direction
                deviation_towards_ai = False
                if cohens_d > 0 and z_score > 0.5:
                    deviation_towards_ai = True
                elif cohens_d < 0 and z_score < -0.5:
                    deviation_towards_ai = True
                    
                explainability_data[feat_name] = {
                    "value": round(val, 4),
                    "human_mean": round(mean, 4),
                    "human_sd": round(sd, 4),
                    "z_score": round(z_score, 2),
                    "deviation_towards_ai": deviation_towards_ai,
                    "cohens_d": round(cohens_d, 2)
                }

    # Map sensitivity to threshold
    sens = getattr(req, 'sensitivity', 'normal').lower()
    threshold = SENSITIVITY_THRESHOLDS.get(sens, SENSITIVITY_THRESHOLDS['normal'])
    if sens not in SENSITIVITY_THRESHOLDS:
        sens = 'normal'

    res = {
        "ai_probability": ai_probability,
        "register": register,
        "register_confidence": register_confidence,
        "is_ai": ai_probability >= threshold,
        "features": features_dict if req.return_features else None,
        "explainability": explainability_data if explainability_data else None,
        "cascade_level": cascade_level,
        "processing_time_ms": round(elapsed_ms, 2),
        "sentences": sentences_data,
        "neural_signals": neural_signals,
        "model_attribution": attribution,
        "is_calibrated": True,
        "sensitivity_applied": sens
    }
    
    set_cache(cache_key, res)
    return res



@app.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(req: BatchDetectRequest, request: Request):
    if not _check_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Limit: 60 req/min.")

    m = get_models()
    t0 = time.time()

    if len(req.texts) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit: 1000 texts.")

    # Map sensitivity to threshold
    sens = getattr(req, 'sensitivity', 'normal').lower()
    threshold = SENSITIVITY_THRESHOLDS.get(sens, SENSITIVITY_THRESHOLDS['normal'])
    if sens not in SENSITIVITY_THRESHOLDS:
        sens = 'normal'

    results = []
    for text in req.texts:
        text_norm = normalize_text_defensive(text)
        feat_vector = extract_feature_vector(text_norm, feature_cols=m['feature_cols'], extended=False)
        if feat_vector is None:
            results.append(DetectResponse(
                ai_probability=0.0,
                register='unknown',
                register_confidence=None,
                is_ai=False,
                features=None,
                processing_time_ms=0.0,
                sensitivity_applied=sens
            ))
            continue

        X = np.array([feat_vector])

        if req.register:
            register = req.register
            if register not in m['detectors']:
                results.append(DetectResponse(
                    ai_probability=0.0,
                    register=register,
                    register_confidence=None,
                    is_ai=False,
                    features=None,
                    processing_time_ms=0.0,
                    sensitivity_applied=sens
                ))
                continue
        else:
            register, _ = classify_register(feat_vector, m)

        # Cascading Classifier Architecture
        detector = m['detectors'].get(register, m.get('all_detector'))
        if detector is None:
            results.append(DetectResponse(
                ai_probability=0.0,
                register=register,
                register_confidence=None,
                is_ai=False,
                features=None,
                processing_time_ms=0.0,
                sensitivity_applied=sens
            ))
            continue

        # Level 1: Fast Stylometrics
        fast_proba = detector.predict_proba(X)[0]
        classes = detector.classes_
        ai_label_idx = list(classes).index(1) if 1 in classes else 1
        fast_probability = float(fast_proba[ai_label_idx])

        ai_probability = None
        cascade_level = None

        if fast_probability <= 0.15 or fast_probability >= 0.85 or not m.get('hybrid_available'):
            ai_probability = fast_probability
            cascade_level = "level_1_stylometrics"
        else:
            try:
                from tool.hybrid_detector import predict_hybrid
                ai_probability = predict_hybrid(text_norm, register=register)
                cascade_level = "level_2_hybrid_ensemble"
            except Exception:
                ai_probability = fast_probability
                cascade_level = "level_1_stylometrics"

        # Calibration: adjust based on word count
        word_count = len(text_norm.split())
        ai_probability = calibrate_probability(ai_probability, word_count)

        # Explainability Data
        explainability_data = {}
        features_dict = None
        feats = extract_feature_vector(text_norm, feature_cols=m['feature_cols'], extended=False)
        if feats:
            features_dict = dict(zip(m['feature_cols'], feats))

        norms = load_feature_norms()
        reg_norms = norms.get(register, norms.get('all', {}))
        
        if features_dict and reg_norms:
            for feat_name, val in features_dict.items():
                if feat_name in reg_norms:
                    mean = reg_norms[feat_name]['human_mean']
                    sd = reg_norms[feat_name]['human_sd']
                    z_score = (val - mean) / sd if sd > 0 else 0.0
                    cohens_d = reg_norms[feat_name]['cohens_d']
                    
                    deviation_towards_ai = False
                    if cohens_d > 0 and z_score > 0.5:
                        deviation_towards_ai = True
                    elif cohens_d < 0 and z_score < -0.5:
                        deviation_towards_ai = True
                        
                    explainability_data[feat_name] = {
                        "value": round(val, 4),
                        "human_mean": round(mean, 4),
                        "human_sd": round(sd, 4),
                        "z_score": round(z_score, 2),
                        "deviation_towards_ai": deviation_towards_ai,
                        "cohens_d": round(cohens_d, 2)
                    }

        results.append(DetectResponse(
            ai_probability=ai_probability,
            register=register,
            register_confidence=None,
            is_ai=ai_probability >= threshold,
            features=features_dict if req.return_features else None,
            explainability=explainability_data if explainability_data else None,
            cascade_level=cascade_level,
            processing_time_ms=0.0,
            sensitivity_applied=sens
        ))

    total_ms = (time.time() - t0) * 1000
    return BatchDetectResponse(
        results=results,
        total_time_ms=round(total_ms, 2),
        texts_processed=len(results),
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
