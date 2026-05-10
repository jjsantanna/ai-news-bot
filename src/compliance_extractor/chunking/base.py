"""Chunk dataclass + Chunker protocol + helpers shared across strategies.

A Chunk is a window into a Document with all the provenance needed to (a)
ground retrieved evidence back to the source via char offsets, and (b)
populate `evidence_quote_page` in the output xlsx via page_start/page_end
derived from the source Document's TextBlocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from compliance_extractor.ingest.base import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    char_start: int
    char_end: int
    n_tokens: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    strategy: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def page_span(doc: Document, char_start: int, char_end: int) -> tuple[int | None, int | None]:
    """Find the page range (inclusive) for a char window into doc.full_text."""
    pages: list[int] = []
    for blk in doc.blocks:
        if blk.page is None:
            continue
        if blk.char_end <= char_start or blk.char_start >= char_end:
            continue
        pages.append(blk.page)
    if not pages:
        return None, None
    return min(pages), max(pages)


def section_for(doc: Document, char_start: int, char_end: int) -> str | None:
    """Return the most-frequent section path overlapping a char window."""
    counts: dict[str, int] = {}
    for blk in doc.blocks:
        if not blk.section:
            continue
        overlap_start = max(blk.char_start, char_start)
        overlap_end = min(blk.char_end, char_end)
        if overlap_end > overlap_start:
            counts[blk.section] = counts.get(blk.section, 0) + (overlap_end - overlap_start)
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]
