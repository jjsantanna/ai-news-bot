"""Map (char_start, char_end) into source-document page/section metadata."""
from __future__ import annotations

from compliance_extractor.ingest.base import Document
from compliance_extractor.chunking.base import page_span, section_for


def locate(doc: Document, char_start: int, char_end: int) -> tuple[int | None, int | None, str | None]:
    """Return (page_start, page_end, section) for a char window in `doc`."""
    ps, pe = page_span(doc, char_start, char_end)
    section = section_for(doc, char_start, char_end)
    return ps, pe, section
