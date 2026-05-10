"""End-to-end pipeline integration test with a fake LLM.

Exercises the full path: load (plaintext) -> chunk (paragraph) -> retrieve
(bm25) -> extract (rag) -> validate (exact rung) -> emit DataFrame. The
LLM is faked so the test runs offline; everything else is the real
production code path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compliance_extractor.config import RunConfig
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    TokenUsage,
)
from compliance_extractor.pipeline import run_pipeline
from compliance_extractor.schemas import EvidenceCols


CORPUS = """# Information Security Policy

## Access Control

Multi-factor authentication is required for all administrators accessing
production systems. Local accounts use SSH keys with a minimum length of
256 bits.

## Backup Strategy

Backups are taken nightly and encrypted at rest using AES-256. Retention
is 90 days for daily snapshots and 1 year for monthly archives.

## Incident Response

Incident response runbooks are maintained in the security wiki and
reviewed quarterly. The on-call rotation includes a primary and a
secondary engineer.
"""


class _FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self):
        self.requests: list[ExtractionRequest] = []

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        self.requests.append(request)
        first_doc = request.documents[0]
        if "authentication" in request.control_description.lower():
            quote = "Multi-factor authentication is required for all administrators"
            decision = "complies"
        elif "backup" in request.control_description.lower():
            quote = "Backups are taken nightly and encrypted at rest using AES-256"
            decision = "complies"
        else:
            quote = ""
            decision = "no-evidence"
        if quote and quote not in first_doc.text:
            for doc in request.documents:
                if quote in doc.text:
                    first_doc = doc
                    break
        return ExtractionResponse(
            decision=decision,
            evidence_quote=quote,
            evidence_quote_source=first_doc.doc_id if quote else "",
            rationale="fake rationale",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            provider=self.name,
            model=self.model,
        )


@pytest.mark.fast
def test_pipeline_end_to_end_rag(tmp_path: Path) -> None:
    pytest.importorskip("rank_bm25")
    pytest.importorskip("openpyxl")

    corpus_path = tmp_path / "policy.md"
    corpus_path.write_text(CORPUS)

    controls_df = pd.DataFrame([
        {"control_id": "A.5.15", "title": "Access Control",
         "description": "Authentication is required for administrators."},
        {"control_id": "A.8.13", "title": "Backups",
         "description": "Backups are encrypted and retained."},
        {"control_id": "A.99.0", "title": "Quantum Crypto",
         "description": "Use quantum-safe ciphers."},
    ])

    config = RunConfig(
        variant_id="v_test_rag",
        description="integration",
        chunker="paragraph",
        chunker_params={"target_tokens": 200, "sentence_backend": "regex"},
        retriever="bm25",
        extractor="rag",
        provider="anthropic",
        llm="fake",
        k=2,
    )

    llm = _FakeLLM()
    df = run_pipeline(config, [corpus_path], controls_df, llm=llm)

    assert len(df) == 3
    assert list(df[EvidenceCols.CONTROL_ID]) == ["A.5.15", "A.8.13", "A.99.0"]

    mfa_row = df[df[EvidenceCols.CONTROL_ID] == "A.5.15"].iloc[0]
    assert mfa_row[EvidenceCols.VALIDATED_EXISTS]
    assert mfa_row[EvidenceCols.MATCH_METHOD] == "exact"
    assert mfa_row[EvidenceCols.PAGE] is pd.NA or pd.isna(mfa_row[EvidenceCols.PAGE]) or mfa_row[EvidenceCols.PAGE] >= 0
    assert mfa_row[EvidenceCols.CHAR_START] >= 0
    assert mfa_row[EvidenceCols.EXTRACTOR] == "rag"

    none_row = df[df[EvidenceCols.CONTROL_ID] == "A.99.0"].iloc[0]
    assert not bool(none_row[EvidenceCols.VALIDATED_EXISTS])
    assert none_row[EvidenceCols.EVIDENCE_TEXT] == ""

    assert all(c in df.columns for c in [
        EvidenceCols.CONTROL_ID, EvidenceCols.DOC_ID, EvidenceCols.EVIDENCE_TEXT,
        EvidenceCols.MATCH_METHOD, EvidenceCols.CONFIDENCE_TIER, EvidenceCols.EXTRACTOR,
    ])


@pytest.mark.fast
def test_pipeline_long_context_skips_chunker_and_retriever(tmp_path: Path) -> None:
    corpus_path = tmp_path / "policy.md"
    corpus_path.write_text(CORPUS)

    controls_df = pd.DataFrame([
        {"control_id": "A.5.15", "title": "Access Control",
         "description": "authentication for administrators"},
    ])

    config = RunConfig(
        variant_id="v_test_lc",
        description="long-context smoke",
        extractor="long_context",
        retriever="none",
        provider="anthropic",
        llm="fake",
    )

    llm = _FakeLLM()
    df = run_pipeline(config, [corpus_path], controls_df, llm=llm)

    assert len(df) == 1
    assert llm.requests[0].documents[0].text.strip() == CORPUS.strip()
    assert df[EvidenceCols.MATCH_METHOD].iloc[0] in {"exact", "normalized"}
