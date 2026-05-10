"""Plaintext / markdown / transcript loader."""
from __future__ import annotations

from pathlib import Path

from compliance_extractor.ingest.base import Document, TextBlock

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log", ".transcript"}


class PlaintextLoader:
    name = "plaintext"

    def can_handle(self, source: str | Path) -> bool:
        p = Path(str(source))
        return p.suffix.lower() in _TEXT_SUFFIXES and p.exists()

    def load(self, source: str | Path) -> Document:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        blocks = [TextBlock(text=text, char_start=0, char_end=len(text), page=None, section=None)]
        return Document(
            doc_id=path.name,
            full_text=text,
            blocks=blocks,
            metadata={"source_path": str(path), "modality": "plaintext"},
        )
