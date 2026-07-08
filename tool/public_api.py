#!/usr/bin/env python3
"""Extended FastAPI app adding public detector endpoints to the base API."""
import os
import time
from typing import List
from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tool.public_detector import predict_ai_probability, MODELS
from tool.api import app as base_app
from tool.api import _check_api_key, check_rate_limit

# Re-export base app with additional endpoints
app = base_app

# Add CORS if not already present (idempotent)
already_has_cors = any(isinstance(m, CORSMiddleware) for m in app.user_middleware)
if not already_has_cors:
    raw_origins = os.environ.get("CORS_ORIGINS", "*")
    allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()] if raw_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials="*" not in allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class PublicDetectRequest(BaseModel):
    text: str = Field(..., description="Text to analyze", max_length=10000)
    detector: str = Field("roberta-openai", description="Public detector name: roberta-openai or chatgpt-detector")
    threshold: float = Field(0.5, description="Decision threshold (0.0 to 1.0)")


class PublicDetectResponse(BaseModel):
    ai_probability: float = Field(..., description="Probability that text is AI-generated")
    detector: str = Field(..., description="Public detector used")
    is_ai: bool = Field(..., description="Binary classification based on threshold")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class PublicDetectorsResponse(BaseModel):
    detectors: List[str] = Field(..., description="Available public detector names")


def _check_public_request(request: Request):
    if not _check_api_key(request):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Limit: 60 req/min.")


@app.get("/public/detectors", response_model=PublicDetectorsResponse)
def list_public_detectors(request: Request):
    _check_public_request(request)
    return PublicDetectorsResponse(detectors=list(MODELS.keys()))


@app.post("/detect/public", response_model=PublicDetectResponse)
def detect_public(req: PublicDetectRequest, request: Request):
    _check_public_request(request)
    if req.detector not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown detector '{req.detector}'. Available: {list(MODELS.keys())}",
        )
    t0 = time.time()
    try:
        prob = predict_ai_probability(req.text, req.detector)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Public detector inference failed: {e}")
    elapsed_ms = (time.time() - t0) * 1000
    return PublicDetectResponse(
        ai_probability=prob,
        detector=req.detector,
        is_ai=prob >= req.threshold,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post("/detect/public/batch", response_model=List[PublicDetectResponse])
def detect_public_batch(reqs: List[PublicDetectRequest], request: Request):
    _check_public_request(request)
    if len(reqs) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit: 100 texts.")
    results = []
    for req in reqs:
        if req.detector not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown detector '{req.detector}'. Available: {list(MODELS.keys())}",
            )
        t0 = time.time()
        try:
            prob = predict_ai_probability(req.text, req.detector)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Public detector inference failed: {e}")
        elapsed_ms = (time.time() - t0) * 1000
        results.append(PublicDetectResponse(
            ai_probability=prob,
            detector=req.detector,
            is_ai=prob >= req.threshold,
            processing_time_ms=round(elapsed_ms, 2),
        ))
    return results


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
