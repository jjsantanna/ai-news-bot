"""diskcache-backed key-value cache for LLM and embedding responses.

Cache key = SHA-256 of (namespace, model, params_json, payload_text). A hit
short-circuits the API call entirely, which makes variant sweeps cheap to
re-run after a code change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import diskcache

_DEFAULT_DIR = Path(os.environ.get("CE_CACHE_DIR", ".cache/ce"))
_cache: diskcache.Cache | None = None


def get_cache(directory: str | Path | None = None) -> diskcache.Cache:
    global _cache
    if _cache is None:
        directory = Path(directory) if directory else _DEFAULT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(directory))
    return _cache


def make_key(namespace: str, model: str, params: dict[str, Any], payload: str) -> str:
    canon = json.dumps(params, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256()
    h.update(namespace.encode())
    h.update(b"\x00")
    h.update(model.encode())
    h.update(b"\x00")
    h.update(canon.encode())
    h.update(b"\x00")
    h.update(payload.encode())
    return h.hexdigest()


def cached_call(namespace: str, model: str, params: dict[str, Any], payload: str, fn):
    """Return cache[key] if present, else compute fn(), store, and return."""
    cache = get_cache()
    key = make_key(namespace, model, params, payload)
    if key in cache:
        return cache[key]
    value = fn()
    cache[key] = value
    return value
