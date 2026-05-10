"""OpenAI-compatible client — covers OpenAI, DeepSeek, Mistral, and Ollama.

All four providers expose a `/v1/chat/completions` endpoint that accepts
the same JSON schema, so a single client with a configurable `base_url`
covers them. None offer a native citations API, so we instruct the model
to return strict JSON (`response_format={'type': 'json_object'}` when
supported) and the downstream validator locates the quote in source.

  - OpenAI:   https://api.openai.com/v1
  - DeepSeek: https://api.deepseek.com/v1
  - Mistral:  https://api.mistral.ai/v1
  - Ollama:   http://localhost:11434/v1
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    TokenUsage,
)
from compliance_extractor.llm.cache import get_cached, make_cache_key, set_cached
from compliance_extractor.llm.parsing import parse_extraction_payload
from compliance_extractor.llm.prompts import SYSTEM_PROMPT, render_user_prompt


_PROVIDER_DEFAULTS = {
    "openai":   {"base_url": "https://api.openai.com/v1",   "model": "gpt-4o-mini",       "json_mode": True},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",     "json_mode": True},
    "mistral":  {"base_url": "https://api.mistral.ai/v1",   "model": "mistral-small-latest", "json_mode": True},
    "ollama":   {"base_url": "http://localhost:11434/v1",   "model": "llama3.1:8b",       "json_mode": False},
}


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        use_cache: bool = True,
        request_extras: dict[str, Any] | None = None,
        json_mode: bool | None = None,
    ) -> None:
        if provider not in _PROVIDER_DEFAULTS:
            raise ValueError(f"Unknown OpenAI-compatible provider: {provider!r}")
        defaults = _PROVIDER_DEFAULTS[provider]
        self.provider = provider
        self.name = provider
        self.model = model or defaults["model"]
        self.base_url = base_url or defaults["base_url"]
        self.api_key = api_key
        self._client = client
        self.use_cache = use_cache
        self.request_extras = request_extras or {}
        self.json_mode = defaults["json_mode"] if json_mode is None else json_mode

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # type: ignore[import-not-found]

            self._client = OpenAI(api_key=self.api_key or "ollama-local", base_url=self.base_url)
        return self._client

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        user_prompt = self._build_user_prompt(request)
        cache_key = make_cache_key(
            provider=self.provider,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            documents=[(d.doc_id, d.text) for d in request.documents],
            extra={"max_tokens": request.max_output_tokens, "json_mode": self.json_mode},
        )
        if self.use_cache:
            hit = get_cached(cache_key)
            if hit is not None:
                return _from_cache_dict(hit, provider=self.provider)

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            **self.request_extras,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        decision, quote, source_id, rationale = parse_extraction_payload(text)
        usage = _extract_usage(response)

        result = ExtractionResponse(
            decision=decision,
            evidence_quote=quote,
            evidence_quote_source=source_id,
            rationale=rationale,
            citation=None,
            usage=usage,
            latency_seconds=latency,
            model=self.model,
            provider=self.provider,
            raw={"response_text": text},
        )
        if self.use_cache:
            set_cached(cache_key, _to_cache_dict(result))
        return result

    @staticmethod
    def _build_user_prompt(request: ExtractionRequest) -> str:
        sections = [render_user_prompt(request), "", "Documents:"]
        for doc in request.documents:
            sections.append(f"--- BEGIN doc_id={doc.doc_id} ---")
            sections.append(doc.text)
            sections.append(f"--- END doc_id={doc.doc_id} ---")
        return "\n".join(sections)


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _to_cache_dict(resp: ExtractionResponse) -> dict[str, Any]:
    return asdict(resp)


def _from_cache_dict(d: dict[str, Any], provider: str) -> ExtractionResponse:
    usage = TokenUsage(**(d.get("usage") or {}))
    return ExtractionResponse(
        decision=d["decision"],
        evidence_quote=d["evidence_quote"],
        evidence_quote_source=d["evidence_quote_source"],
        rationale=d.get("rationale", ""),
        citation=None,
        usage=usage,
        latency_seconds=float(d.get("latency_seconds", 0.0)),
        model=d.get("model", ""),
        provider=d.get("provider", provider),
        raw=d.get("raw", {}),
    )
