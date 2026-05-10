"""Tests for extractors with a fake LLM client.

The fake client just records the request and returns a canned response,
so we can verify each extractor builds the request correctly (which
documents it sends, how many chunks, etc.) without API calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from compliance_extractor.chunking.fixed_token import FixedTokenChunker
from compliance_extractor.extractors.base import Control, ExtractorContext
from compliance_extractor.extractors.citations_rag import CitationsRAGExtractor
from compliance_extractor.extractors.long_context import LongContextExtractor
from compliance_extractor.extractors.mapreduce import MapReduceExtractor
from compliance_extractor.extractors.rag import RAGExtractor
from compliance_extractor.extractors.registry import build_extractor, list_strategies
from compliance_extractor.ingest.base import Document, TextBlock
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    TokenUsage,
)
from compliance_extractor.retrieval.bm25 import BM25Retriever


CORPUS_TEXT = (
    "Multi-factor authentication is required for all administrators on production systems.\n\n"
    "Backups are taken nightly and encrypted at rest using AES-256.\n\n"
    "Incident response runbooks are maintained in the security wiki.\n\n"
    "Access reviews are performed quarterly for privileged users.\n\n"
    "Password complexity is enforced with a minimum of twelve characters."
)


@dataclass
class FakeLLM:
    name: str = "fake"
    model: str = "fake-1"
    decision: str = "complies"
    quote: str = "Multi-factor authentication is required"
    source: str = ""

    def __post_init__(self):
        self.requests: list[ExtractionRequest] = []

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        self.requests.append(request)
        source_id = self.source or (request.documents[0].doc_id if request.documents else "")
        return ExtractionResponse(
            decision=self.decision,
            evidence_quote=self.quote,
            evidence_quote_source=source_id,
            rationale="fake rationale",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            model=self.model,
            provider=self.name,
        )


def _doc(text: str = CORPUS_TEXT, doc_id: str = "policy.txt") -> Document:
    return Document(
        doc_id=doc_id,
        full_text=text,
        blocks=[TextBlock(text, 0, len(text), page=1, section="Security")],
    )


@pytest.mark.fast
def test_long_context_passes_full_documents() -> None:
    llm = FakeLLM()
    ctx = ExtractorContext(
        control=Control("A.5.15", "Access Control", "Restrict access via authentication."),
        documents=[_doc()],
        llm=llm,
    )
    response = LongContextExtractor().extract(ctx)
    assert response.decision == "complies"
    assert len(llm.requests) == 1
    assert len(llm.requests[0].documents) == 1
    assert llm.requests[0].documents[0].text == CORPUS_TEXT
    assert response.raw["extractor"] == "long_context"


@pytest.mark.fast
def test_rag_chunks_and_retrieves_top_k() -> None:
    pytest.importorskip("rank_bm25")
    llm = FakeLLM()
    chunker = FixedTokenChunker(target_tokens=20, overlap_tokens=0, sentence_backend="regex")
    retriever = BM25Retriever()
    ctx = ExtractorContext(
        control=Control("A.5.15", "Access Control", "authentication administrators"),
        documents=[_doc()],
        llm=llm,
        chunker=chunker,
        retriever=retriever,
        top_k=2,
    )
    response = RAGExtractor().extract(ctx)
    assert response.decision == "complies"
    assert len(llm.requests) == 1
    docs_sent = llm.requests[0].documents
    assert 0 < len(docs_sent) <= 2
    assert all(d.doc_id.startswith("policy.txt#") for d in docs_sent)


@pytest.mark.fast
def test_rag_falls_back_to_full_docs_when_retrieval_empty() -> None:
    pytest.importorskip("rank_bm25")
    llm = FakeLLM()
    chunker = FixedTokenChunker(target_tokens=20, overlap_tokens=0, sentence_backend="regex")
    retriever = BM25Retriever()
    ctx = ExtractorContext(
        control=Control("X", "X", "zzz_no_overlap_zzz"),
        documents=[_doc()],
        llm=llm,
        chunker=chunker,
        retriever=retriever,
        top_k=2,
    )
    response = RAGExtractor().extract(ctx)
    docs_sent = llm.requests[0].documents
    assert len(docs_sent) == 1
    assert docs_sent[0].doc_id == "policy.txt"


@pytest.mark.fast
def test_mapreduce_calls_llm_per_chunk_and_picks_best() -> None:
    chunker = FixedTokenChunker(target_tokens=20, overlap_tokens=0, sentence_backend="regex")

    class CycleLLM:
        name = "cycle"
        model = "cycle-1"

        def __init__(self):
            self.calls = 0

        def extract(self, request: ExtractionRequest) -> ExtractionResponse:
            self.calls += 1
            decision = "complies" if self.calls == 2 else "no-evidence"
            return ExtractionResponse(
                decision=decision,
                evidence_quote="MFA is required" if decision == "complies" else "",
                evidence_quote_source=request.documents[0].doc_id if decision == "complies" else "",
                rationale="r",
                usage=TokenUsage(input_tokens=50, output_tokens=10),
                provider=self.name,
                model=self.model,
            )

    llm = CycleLLM()
    ctx = ExtractorContext(
        control=Control("A", "A", "a"),
        documents=[_doc()],
        llm=llm,
        chunker=chunker,
    )
    response = MapReduceExtractor().extract(ctx)
    assert response.decision == "complies"
    assert response.evidence_quote == "MFA is required"
    assert llm.calls > 1
    assert response.usage.input_tokens >= 50 * llm.calls


@pytest.mark.fast
def test_citations_rag_requires_anthropic_client() -> None:
    pytest.importorskip("rank_bm25")
    llm = FakeLLM()
    chunker = FixedTokenChunker(target_tokens=20, overlap_tokens=0, sentence_backend="regex")
    retriever = BM25Retriever()
    ctx = ExtractorContext(
        control=Control("A", "A", "a"),
        documents=[_doc()],
        llm=llm,
        chunker=chunker,
        retriever=retriever,
    )
    with pytest.raises(ValueError, match="citations_rag requires AnthropicClient"):
        CitationsRAGExtractor().extract(ctx)


@pytest.mark.fast
def test_registry_dispatch_and_listing() -> None:
    assert set(list_strategies()) == {"rag", "long_context", "mapreduce", "citations_rag"}
    e = build_extractor("rag")
    assert e.name == "rag"
    with pytest.raises(ValueError, match="Unknown extractor"):
        build_extractor("nope")
