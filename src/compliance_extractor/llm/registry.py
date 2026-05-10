"""LLM client factory: build an LLMClient from variant config provider+model."""
from __future__ import annotations

from typing import Any

from compliance_extractor.llm.anthropic_client import AnthropicClient
from compliance_extractor.llm.base import LLMClient
from compliance_extractor.llm.openai_compatible import OpenAICompatibleClient

_OPENAI_COMPAT = {"openai", "deepseek", "mistral", "ollama"}


def build_client(provider: str, **kwargs: Any) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    if provider in _OPENAI_COMPAT:
        return OpenAICompatibleClient(provider=provider, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def list_providers() -> list[str]:
    return ["anthropic", "openai", "deepseek", "mistral", "ollama"]
