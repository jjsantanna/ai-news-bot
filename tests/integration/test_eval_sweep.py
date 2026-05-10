"""Integration test: sweep two variants over one corpus + gold set.

Exercises the full eval grid path in miniature: define a small
controls + gold + corpus fixture, run two variants (RAG vs long
context) with a deterministic fake LLM, and verify the sweep DataFrame
has both rows with sane precision/recall/F1.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compliance_extractor.config import RunConfig
from compliance_extractor.eval.matcher import match, quote_similarity
from compliance_extractor.eval.metrics import Metrics, evaluate
from compliance_extractor.eval.sweep import run_sweep
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    TokenUsage,
)


CORPUS = (
    "Multi-factor authentication is required for all administrators on production systems.\n\n"
    "Backups are taken nightly and encrypted at rest using AES-256.\n\n"
    "Incident response runbooks are maintained in the security wiki."
)


class _TopicalLLM:
    name = "fake"
    model = "fake-1"

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        cd = request.control_description.lower()
        target_doc = request.documents[0]
        if "authentication" in cd:
            quote = "Multi-factor authentication is required for all administrators on production systems"
        elif "backup" in cd:
            quote = "Backups are taken nightly and encrypted at rest using AES-256"
        else:
            return ExtractionResponse(
                decision="no-evidence",
                evidence_quote="",
                evidence_quote_source="",
                rationale="not relevant",
                provider=self.name,
                model=self.model,
                usage=TokenUsage(input_tokens=80, output_tokens=10),
            )
        for doc in request.documents:
            if quote in doc.text:
                target_doc = doc
                break
        return ExtractionResponse(
            decision="complies",
            evidence_quote=quote,
            evidence_quote_source=target_doc.doc_id,
            rationale="found",
            usage=TokenUsage(input_tokens=120, output_tokens=20),
            provider=self.name,
            model=self.model,
        )


@pytest.mark.fast
def test_quote_similarity_and_match() -> None:
    pytest.importorskip("rapidfuzz")
    sim = quote_similarity("Multi-factor authentication required", "Multi-factor authentication is required")
    assert 0.7 <= sim <= 1.0
    res = match("AAA BBB CCC", "AAA BBB CCC", threshold=0.9)
    assert res.is_match
    assert match("totally different", "Multi-factor", threshold=0.9).is_match is False


@pytest.mark.fast
def test_metrics_perfect_match() -> None:
    pytest.importorskip("rapidfuzz")
    preds = pd.DataFrame([
        {"control_id": "A.5.15", "evidence_text": "Multi-factor authentication is required",
         "validated_exists": True, "match_method": "exact"},
        {"control_id": "A.8.13", "evidence_text": "Backups are taken nightly",
         "validated_exists": True, "match_method": "exact"},
    ])
    gold = pd.DataFrame([
        {"control_id": "A.5.15", "exact_quote": "Multi-factor authentication is required"},
        {"control_id": "A.8.13", "exact_quote": "Backups are taken nightly"},
    ])
    m = evaluate(preds, gold)
    assert m.tp == 2 and m.fp == 0 and m.fn == 0
    assert m.f1 == 1.0
    assert m.hallucination_rate == 0.0


@pytest.mark.fast
def test_metrics_detects_hallucination() -> None:
    pytest.importorskip("rapidfuzz")
    preds = pd.DataFrame([
        {"control_id": "A.5.15", "evidence_text": "Quantum cryptography enforces zero-trust",
         "validated_exists": False, "match_method": "none"},
    ])
    gold = pd.DataFrame([
        {"control_id": "A.5.15", "exact_quote": "Multi-factor authentication is required"},
    ])
    m = evaluate(preds, gold)
    assert m.hallucinations == 1
    assert m.hallucination_rate == 1.0
    assert m.tp == 0


@pytest.mark.fast
def test_run_sweep_two_variants(tmp_path: Path) -> None:
    pytest.importorskip("rapidfuzz")
    pytest.importorskip("rank_bm25")

    corpus_path = tmp_path / "policy.md"
    corpus_path.write_text(CORPUS)

    controls = pd.DataFrame([
        {"control_id": "A.5.15", "title": "Access Control",
         "description": "authentication is required for administrators"},
        {"control_id": "A.8.13", "title": "Backups",
         "description": "backups encrypted at rest"},
    ])
    gold = pd.DataFrame([
        {"control_id": "A.5.15",
         "exact_quote": "Multi-factor authentication is required for all administrators"},
        {"control_id": "A.8.13",
         "exact_quote": "Backups are taken nightly and encrypted at rest using AES-256"},
    ])

    variant_rag = RunConfig(
        variant_id="v_rag",
        chunker="paragraph",
        chunker_params={"target_tokens": 120, "sentence_backend": "regex"},
        retriever="bm25",
        extractor="rag",
        provider="anthropic",
        llm="fake",
        k=2,
    )
    variant_lc = RunConfig(
        variant_id="v_long_context",
        retriever="none",
        extractor="long_context",
        provider="anthropic",
        llm="fake",
    )

    llm_factory = lambda _cfg: _TopicalLLM()
    df = run_sweep([variant_rag, variant_lc], [corpus_path], controls, gold, llm_factory=llm_factory)

    assert set(df["variant_id"]) == {"v_rag", "v_long_context"}
    assert all(df["f1"] >= 0.8)
    assert all(df["hallucination_rate"] == 0.0)
