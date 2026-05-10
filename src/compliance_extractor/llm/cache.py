"""Hash-keyed cache wrapper for LLM responses.

The cache key includes the provider, model, system prompt, user prompt, and
documents (by content hash + order), so any change in any input invalidates
the entry. This is the same caching contract Anthropic prompt caching uses
on the wire — plus a local-disk hit so re-running a sweep variant doesn't
re-pay for identical calls.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from compliance_extractor.utils.caching import get_cache


def make_cache_key(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    documents: list[tuple[str, str]],
    extra: dict[str, Any] | None = None,
) -> str:
    parts: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "system": _sha(system_prompt),
        "user": _sha(user_prompt),
        "documents": [(doc_id, _sha(text)) for doc_id, text in documents],
        "extra": extra or {},
    }
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get_cached(key: str) -> Any | None:
    return get_cache().get(f"llm:{key}")


def set_cached(key: str, value: Any, expire: int | None = None) -> None:
    get_cache().set(f"llm:{key}", value, expire=expire)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
