#!/usr/bin/env python3
"""Shared API security utilities: caching, rate limiting, and API-key checks."""
import os
import time
import hashlib
from typing import Optional, Dict, Any

from fastapi import Request

# ── In-Memory Cache ────────────────────────────────────────────────────────

RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}
DEFAULT_CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_ENTRIES = int(os.environ.get("MAX_CACHE_ENTRIES", "1000"))


def get_cache_key(text: str, register: Optional[str]) -> str:
    raw_key = f"{text}||{register or ''}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()


def _is_stale(entry: dict) -> bool:
    return (time.time() - entry["ts"]) > DEFAULT_CACHE_TTL_SECONDS


def get_cache(key: str):
    """Return a copy of the cached data if it exists and is not stale, else None."""
    entry = RESPONSE_CACHE.get(key)
    if entry is None or _is_stale(entry):
        return None
    return entry["data"].copy()


def set_cache(key: str, data: dict):
    # Evict stale entries first
    stale_keys = [k for k, v in RESPONSE_CACHE.items() if _is_stale(v)]
    for k in stale_keys:
        RESPONSE_CACHE.pop(k, None)
    if len(RESPONSE_CACHE) >= MAX_CACHE_ENTRIES:
        first_key = next(iter(RESPONSE_CACHE))
        RESPONSE_CACHE.pop(first_key)
    RESPONSE_CACHE[key] = {"data": data, "ts": time.time()}


# ── Rate Limiting ──────────────────────────────────────────────────────────

RATE_LIMITS: Dict[str, Dict[str, float]] = {}
RATE_LIMIT_MAX_IPS = int(os.environ.get("RATE_LIMIT_MAX_IPS", "10000"))
RATE_LIMIT_IP_TTL_SECONDS = float(os.environ.get("RATE_LIMIT_IP_TTL_SECONDS", "300"))
API_KEY = os.environ.get("API_KEY", "")


def _check_api_key(request: Request) -> bool:
    if not API_KEY:
        return True
    header_key = request.headers.get("x-api-key", "")
    return header_key == API_KEY


def _evict_stale_rate_limits(now: float):
    """Evict rate-limit entries that have been inactive beyond the TTL."""
    stale = [ip for ip, state in RATE_LIMITS.items() if (now - state["last_updated"]) > RATE_LIMIT_IP_TTL_SECONDS]
    for ip in stale:
        RATE_LIMITS.pop(ip, None)


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    limit = 60.0  # max tokens
    refill_rate = 1.0  # 1 token per second

    # Evict stale entries to bound memory usage
    _evict_stale_rate_limits(now)

    # Bound total number of tracked IPs
    if ip not in RATE_LIMITS and len(RATE_LIMITS) >= RATE_LIMIT_MAX_IPS:
        return False

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
