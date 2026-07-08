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
import hashlib
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tool.feature_extractor import (
    extract_feature_vector,
    normalize_unicode,
    ORIGINAL_FEATURE_COLS,
    ALL_FEATURE_COLS,
)

# New module imports
from tool.neural_detector import compute_perplexity_and_burstiness
from tool.sentence_analyzer import analyze_sentences
from tool.attribution import attribute_source
from tool.adversarial_defense import normalize_text_defensive
from tool.calibration import calibrate_probability

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# ── Model loading ──────────────────────────────────────────────────────────

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

    # Check if SOTA hybrid ensemblers are available
    hybrid_path = os.path.join(MODELS_DIR, 'beemo_register_ensemblers.joblib')
    onnx_path = os.path.join(MODELS_DIR, 'deberta_onnx_quantized.onnx')
    if os.path.exists(hybrid_path) and os.path.exists(onnx_path):
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

# CORS configuration
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


# ── In-Memory Cache and Rate Limiting ──────────────────────────────────────

RESPONSE_CACHE = {}  # key (hash) -> {"data": dict, "ts": float}
DEFAULT_CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ENTRIES = int(os.environ.get("MAX_CACHE_ENTRIES", "1000"))

def get_cache_key(text: str, register: Optional[str]) -> str:
    raw_key = f"{text}||{register or ''}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()

def _is_stale(entry: dict) -> bool:
    return (time.time() - entry["ts"]) > DEFAULT_CACHE_TTL_SECONDS

def set_cache(key: str, data: dict):
    # Evict stale entries first
    stale_keys = [k for k, v in RESPONSE_CACHE.items() if _is_stale(v)]
    for k in stale_keys:
        RESPONSE_CACHE.pop(k, None)
    if len(RESPONSE_CACHE) >= MAX_CACHE_ENTRIES:
        first_key = next(iter(RESPONSE_CACHE))
        RESPONSE_CACHE.pop(first_key)
    RESPONSE_CACHE[key] = {"data": data, "ts": time.time()}

RATE_LIMITS = {}  # ip -> {"tokens": float, "last_updated": float}
API_KEY = os.environ.get("API_KEY", "")

def _check_api_key(request: Request):
    if not API_KEY:
        return True
    header_key = request.headers.get("x-api-key", "")
    return header_key == API_KEY

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    limit = 60.0  # max tokens
    refill_rate = 1.0  # 1 token per second
    
    if ip not in RATE_LIMITS:
        RATE_LIMITS[ip] = {"tokens": limit, "last_updated": now}
        return True
        
    state = RATE_LIMITS[ip]
    elapsed = now - state["last_updated"]
    tokens = min(limit, state["tokens"] + elapsed * refill_rate)
    
    if tokens < 1.0:
        return False
        
    RATE_LIMITS[ip] = {"tokens": tokens - 1.0, "last_updated": now}
    return True


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
    if cache_key in RESPONSE_CACHE and not _is_stale(RESPONSE_CACHE[cache_key]):
        cached = RESPONSE_CACHE[cache_key]["data"].copy()
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
        if 'register_classifier' in m:
            reg_pred = m['register_classifier'].predict(X)[0]
            reg_proba = m['register_classifier'].predict_proba(X)[0]
            classes = m['register_classifier'].classes_
            idx = list(classes).index(reg_pred)
            register_confidence = float(reg_proba[idx])
            # Decode label encoder if available
            if 'register_label_encoder' in m:
                register = str(m['register_label_encoder'].inverse_transform([reg_pred])[0])
            else:
                register = str(reg_pred)
        else:
            register = 'all'
            register_confidence = None

    # Step 4: Per-register detection / Hybrid ensembler
    if register in m['detectors']:
        detector = m['detectors'][register]
    elif 'all_detector' in m:
        detector = m['all_detector']
        register = 'all'
    else:
        detector = None

    ai_probability = None
    if m.get('hybrid_available'):
        try:
            from tool.hybrid_detector import predict_hybrid
            ai_probability = predict_hybrid(text, register=register)
        except Exception as e:
            print(f"Error executing hybrid prediction: {e}. Falling back to standard classifier.")

    if ai_probability is None:
        if detector is None:
            raise HTTPException(status_code=500, detail="No detector available.")
        ai_proba = detector.predict_proba(X)[0]
        classes = detector.classes_
        ai_label_idx = list(classes).index(1) if 1 in classes else 1
        ai_probability = float(ai_proba[ai_label_idx])

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
    if req.return_features:
        feats = extract_feature_vector(text, feature_cols=m['feature_cols'], extended=False)
        features_dict = dict(zip(m['feature_cols'], feats)) if feats else None

    # Map sensitivity to threshold
    sens = getattr(req, 'sensitivity', 'normal').lower()
    if sens == 'high':
        threshold = 0.35
    elif sens == 'low':
        threshold = 0.70
    else:
        threshold = 0.50
        sens = 'normal'

    res = {
        "ai_probability": ai_probability,
        "register": register,
        "register_confidence": register_confidence,
        "is_ai": ai_probability >= threshold,
        "features": features_dict,
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
    if sens == 'high':
        threshold = 0.35
    elif sens == 'low':
        threshold = 0.70
    else:
        threshold = 0.50
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
        elif 'register_classifier' in m:
            reg_pred = m['register_classifier'].predict(X)[0]
            if 'register_label_encoder' in m:
                register = str(m['register_label_encoder'].inverse_transform([reg_pred])[0])
            else:
                register = str(reg_pred)
        else:
            register = 'all'

        ai_probability = None
        if m.get('hybrid_available'):
            try:
                from tool.hybrid_detector import predict_hybrid
                ai_probability = predict_hybrid(text_norm, register=register)
            except Exception:
                pass

        if ai_probability is None:
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

            ai_proba = detector.predict_proba(X)[0]
            classes = detector.classes_
            ai_label_idx = list(classes).index(1) if 1 in classes else 1
            ai_probability = float(ai_proba[ai_label_idx])

        # Calibration: adjust based on word count
        word_count = len(text_norm.split())
        ai_probability = calibrate_probability(ai_probability, word_count)

        results.append(DetectResponse(
            ai_probability=ai_probability,
            register=register,
            register_confidence=None,
            is_ai=ai_probability >= threshold,
            features=None,
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
