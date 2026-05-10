"""Citations-RAG extractor: same as RAG but only valid for Anthropic.

Identical retrieval path to RAG; the difference is policy — this
extractor refuses to run unless the LLM client is the Anthropic client
with citations enabled, so the variant config can guarantee that runs
labeled `citations_rag` always produce native citations the validator
short-circuits on.
"""
from __future__ import annotations

from compliance_extractor.extractors.base import ExtractorContext
from compliance_extractor.extractors.rag import RAGExtractor
from compliance_extractor.llm.anthropic_client import AnthropicClient
from compliance_extractor.llm.base import ExtractionResponse


class CitationsRAGExtractor:
    name = "citations_rag"

    def __init__(self) -> None:
        self._inner = RAGExtractor()

    def extract(self, ctx: ExtractorContext) -> ExtractionResponse:
        if not isinstance(ctx.llm, AnthropicClient):
            raise ValueError(
                f"citations_rag requires AnthropicClient, got {type(ctx.llm).__name__}"
            )
        response = self._inner.extract(ctx)
        response.raw = {**response.raw, "extractor": self.name}
        return response
