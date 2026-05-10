"""Build an LLMClient from a RunConfig (provider + model + params)."""
from __future__ import annotations

from compliance_extractor.config import RunConfig
from compliance_extractor.llm.base import LLMClient
from compliance_extractor.llm.registry import build_client


def build_llm_from_config(config: RunConfig) -> LLMClient:
    params = dict(config.llm_params)
    return build_client(config.provider, model=config.llm, **params)
