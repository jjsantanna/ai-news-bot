"""Tests for the validation ladder.

Covers each rung end-to-end on a small corpus, plus the citation
short-circuit, the missing-source path, the no-evidence skip, and the
semantic rung via an injected fake score function so the test suite
needs no model download.
"""
from __future__ import annotations

import pytest

from compliance_extractor.ingest.base import Document, TextBlock
from compliance_extractor.llm.base import CitedSpan
from compliance_extractor.validation.ladder import Validator, ValidatorConfig
from compliance_extractor.validation.normalize import normalize_for_match


def _doc(text: str, doc_id: str = "policy.txt") -> Document:
    return Document(
        doc_id=doc_id,
        full_text=text,
        blocks=[TextBlock(text, 0, len(text), page=1, section="Security")],
    )


@pytest.mark.fast
def test_normalize_collapses_whitespace_and_smart_quotes() -> None:
    src = "  Multi-factor   authentication is “required”.  "
    assert normalize_for_match(src) == 'multi-factor authentication is "required".'


@pytest.mark.fast
def test_validator_no_evidence_short_circuits() -> None:
    v = Validator()
    res = v.validate("", "", [_doc("anything")])
    assert res.status == "no_evidence"
    assert res.match_method == "skipped"


@pytest.mark.fast
def test_validator_exact_match() -> None:
    text = "Multi-factor authentication is required for all administrators."
    v = Validator()
    res = v.validate("Multi-factor authentication", "policy.txt", [_doc(text)])
    assert res.status == "validated"
    assert res.match_method == "exact"
    assert res.located_char_start == 0
    assert res.located_char_end == len("Multi-factor authentication")
    assert res.located_page_start == 1
    assert res.located_section == "Security"


@pytest.mark.fast
def test_validator_normalized_match_handles_whitespace_and_case() -> None:
    text = "Multi-factor authentication is required for all administrators."
    v = Validator()
    res = v.validate("multi-factor    AUTHENTICATION is required", "policy.txt", [_doc(text)])
    assert res.status == "validated"
    assert res.match_method == "normalized"


@pytest.mark.fast
def test_validator_fuzzy_match_above_threshold() -> None:
    pytest.importorskip("rapidfuzz")
    text = "Multi-factor authentication is required for all administrators."
    v = Validator(ValidatorConfig(fuzzy_threshold=80.0))
    res = v.validate("Multi-factor authentication required for administrators", "policy.txt", [_doc(text)])
    assert res.status == "validated"
    assert res.match_method == "fuzzy"
    assert 0.8 <= res.similarity_score <= 1.0


@pytest.mark.fast
def test_validator_hallucinated_when_quote_absent() -> None:
    text = "Backups are encrypted at rest."
    v = Validator(ValidatorConfig(fuzzy_threshold=99.0))
    res = v.validate("Quantum cryptography enforces zero-trust networking.", "policy.txt", [_doc(text)])
    assert res.status == "hallucinated"
    assert res.match_method == "none"


@pytest.mark.fast
def test_validator_semantic_rung_via_fake_score() -> None:
    text = "All admin accounts use MFA tokens."
    fake_score = lambda q, t: 0.9
    v = Validator(ValidatorConfig(enable_semantic=True, semantic_score_fn=fake_score, fuzzy_threshold=99.0))
    res = v.validate("Administrators must authenticate with two factors.", "policy.txt", [_doc(text)])
    assert res.status == "semantic"
    assert res.match_method == "semantic"
    assert res.similarity_score == 0.9


@pytest.mark.fast
def test_validator_missing_source_when_doc_id_unknown() -> None:
    v = Validator()
    res = v.validate("anything", "no_such_doc.txt", [_doc("text", doc_id="actual.txt")])
    assert res.status == "missing_source"


@pytest.mark.fast
def test_validator_uses_anthropic_citation_when_provided() -> None:
    text = "Multi-factor authentication is required."
    citation = CitedSpan(doc_id="policy.txt", char_start=0, char_end=27, cited_text="Multi-factor authentication")
    v = Validator()
    res = v.validate("Multi-factor authentication", "policy.txt", [_doc(text)], citation=citation)
    assert res.status == "validated"
    assert res.match_method == "citation"
    assert res.located_char_start == 0
    assert res.located_char_end == 27


@pytest.mark.fast
def test_validator_falls_back_when_only_one_doc_and_source_id_blank() -> None:
    text = "Backup retention is 90 days."
    v = Validator()
    res = v.validate("Backup retention is 90 days.", "", [_doc(text)])
    assert res.status == "validated"
    assert res.match_method == "exact"
