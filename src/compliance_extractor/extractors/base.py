"""Extractor protocol and shared types.

Extractors orchestrate the chunking → retrieval → LLM pipeline for one
control against one corpus. Each extractor is a different research
strategy (RAG, long-context, map-reduce, citations-RAG); they all return
the same ExtractionResponse so the validator and IO layer don't care
which strategy ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from compliance_extractor.chunking.base import Chunker
from compliance_extractor.ingest.base import Document
from compliance_extractor.llm.base import ExtractionResponse, LLMClient
from compliance_extractor.retrieval.base import Retriever


@dataclass(frozen=True)
class Control:
    control_id: str
    title: str
    description: str


@dataclass
class ExtractorContext:
    control: Control
    documents: list[Document]
    llm: LLMClient
    chunker: Chunker | None = None
    retriever: Retriever | None = None
    top_k: int = 5
    max_output_tokens: int = 1024
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, ctx: ExtractorContext) -> ExtractionResponse: ...
