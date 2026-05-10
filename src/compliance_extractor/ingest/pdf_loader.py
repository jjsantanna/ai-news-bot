"""PDF loader using pypdf. Preserves per-page text + char offsets in full_text."""
from __future__ import annotations

from pathlib import Path

from compliance_extractor.ingest.base import Document, TextBlock


class PdfLoader:
    name = "pdf"

    def can_handle(self, source: str | Path) -> bool:
        p = Path(str(source))
        return p.suffix.lower() == ".pdf" and p.exists()

    def load(self, source: str | Path) -> Document:
        from pypdf import PdfReader  # imported lazily so the dep can be optional in tests

        path = Path(source)
        reader = PdfReader(str(path))
        blocks: list[TextBlock] = []
        full_parts: list[str] = []
        offset = 0
        for i, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.endswith("\n"):
                page_text = page_text + "\n"
            blocks.append(
                TextBlock(
                    text=page_text,
                    char_start=offset,
                    char_end=offset + len(page_text),
                    page=i,
                    section=None,
                )
            )
            full_parts.append(page_text)
            offset += len(page_text)
        full_text = "".join(full_parts)
        return Document(
            doc_id=path.name,
            full_text=full_text,
            blocks=blocks,
            metadata={"source_path": str(path), "modality": "pdf", "n_pages": len(reader.pages)},
        )
