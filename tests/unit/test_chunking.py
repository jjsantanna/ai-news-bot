"""Tests for chunkers: char-offset roundtrip, token-budget compliance, registry."""
from __future__ import annotations

import pytest

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.chunking.fixed_token import FixedTokenChunker
from compliance_extractor.chunking.paragraph import ParagraphChunker
from compliance_extractor.chunking.registry import build_chunker, list_strategies
from compliance_extractor.chunking.section import SectionChunker
from compliance_extractor.chunking.sentence_splitter import split_sentences
from compliance_extractor.chunking.tokens import count_tokens
from compliance_extractor.ingest.base import Document, TextBlock


def _doc(text: str, blocks: list[TextBlock] | None = None, doc_id: str = "d.txt") -> Document:
    if blocks is None:
        blocks = [TextBlock(text, 0, len(text))]
    return Document(doc_id=doc_id, full_text=text, blocks=blocks)


@pytest.mark.fast
def test_count_tokens_basic() -> None:
    pytest.importorskip("tiktoken")
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0


@pytest.mark.fast
def test_split_sentences_regex_fallback_offsets() -> None:
    text = "First sentence. Second one! Third? Done."
    parts = split_sentences(text, backend="regex")
    assert len(parts) >= 3
    for s, cs, ce in parts:
        assert text[cs:ce] == s


@pytest.mark.fast
def test_fixed_token_respects_target_and_offsets() -> None:
    pytest.importorskip("tiktoken")
    text = ("Sentence A. " * 100).strip()
    doc = _doc(text)
    ftc = FixedTokenChunker(target_tokens=50, overlap_tokens=0, sentence_backend="regex")
    chunks = ftc.chunk(doc)
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        assert text[c.char_start:c.char_end] == c.text
        assert c.n_tokens <= 50 or c.text.count(".") == 1
        assert c.strategy == "fixed_token"


@pytest.mark.fast
def test_fixed_token_overlap_resamples_tail() -> None:
    pytest.importorskip("tiktoken")
    text = ". ".join(f"S{i}" for i in range(60)) + "."
    doc = _doc(text)
    ftc_no = FixedTokenChunker(target_tokens=20, overlap_tokens=0, sentence_backend="regex")
    ftc_yes = FixedTokenChunker(target_tokens=20, overlap_tokens=8, sentence_backend="regex")
    n_no = len(ftc_no.chunk(doc))
    n_yes = len(ftc_yes.chunk(doc))
    assert n_yes >= n_no  # overlap can only add re-emitted material


@pytest.mark.fast
def test_paragraph_chunker_packs_paragraphs() -> None:
    pytest.importorskip("tiktoken")
    paras = [f"Paragraph number {i}. It has some text in it." for i in range(8)]
    text = "\n\n".join(paras)
    doc = _doc(text)
    pc = ParagraphChunker(target_tokens=40, sentence_backend="regex")
    chunks = pc.chunk(doc)
    assert chunks
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text
        assert c.strategy == "paragraph"


@pytest.mark.fast
def test_section_chunker_groups_by_section() -> None:
    pytest.importorskip("tiktoken")
    seg_a = "Alpha section content. " * 5
    seg_b = "Bravo section content. " * 5
    text = seg_a + seg_b
    blocks = [
        TextBlock(seg_a, 0, len(seg_a), section="A"),
        TextBlock(seg_b, len(seg_a), len(seg_a) + len(seg_b), section="B"),
    ]
    doc = _doc(text, blocks=blocks)
    sc = SectionChunker(target_tokens=200, sentence_backend="regex")
    chunks = sc.chunk(doc)
    sections = {c.section for c in chunks}
    assert sections == {"A", "B"}
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text


@pytest.mark.fast
def test_section_chunker_falls_back_when_no_sections() -> None:
    pytest.importorskip("tiktoken")
    text = "Para A.\n\nPara B.\n\nPara C."
    doc = _doc(text)
    sc = SectionChunker(target_tokens=20, sentence_backend="regex")
    chunks = sc.chunk(doc)
    assert chunks


@pytest.mark.fast
def test_registry_dispatch_and_listing() -> None:
    assert set(list_strategies()) == {"fixed_token", "paragraph", "section", "semantic"}
    pytest.importorskip("tiktoken")
    chunker = build_chunker("fixed_token", target_tokens=100, overlap_tokens=0, sentence_backend="regex")
    assert chunker.name == "fixed_token"
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        build_chunker("nonexistent_strategy")


@pytest.mark.fast
def test_page_span_and_section_for() -> None:
    from compliance_extractor.chunking.base import page_span, section_for

    text = "AAA\nBBB\nCCC\n"
    blocks = [
        TextBlock("AAA\n", 0, 4, page=1, section="S1"),
        TextBlock("BBB\n", 4, 8, page=2, section="S1"),
        TextBlock("CCC\n", 8, 12, page=3, section="S2"),
    ]
    doc = Document(doc_id="d", full_text=text, blocks=blocks)
    assert page_span(doc, 0, 4) == (1, 1)
    assert page_span(doc, 0, 12) == (1, 3)
    assert section_for(doc, 0, 8) == "S1"
    assert section_for(doc, 5, 12) in {"S1", "S2"}
