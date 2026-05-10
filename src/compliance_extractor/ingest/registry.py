"""Loader registry: dispatch a source string/path to the right loader."""
from __future__ import annotations

from pathlib import Path

from compliance_extractor.ingest.base import Document, DocumentLoader
from compliance_extractor.ingest.docx_loader import DocxLoader
from compliance_extractor.ingest.pdf_loader import PdfLoader
from compliance_extractor.ingest.plaintext_loader import PlaintextLoader
from compliance_extractor.ingest.url_loader import UrlLoader

_REGISTRY: list[DocumentLoader] = [
    UrlLoader(),
    PdfLoader(),
    DocxLoader(),
    PlaintextLoader(),
]


def get_loader(source: str | Path) -> DocumentLoader:
    """Return the first registered loader that can handle `source`."""
    for loader in _REGISTRY:
        if loader.can_handle(source):
            return loader
    raise ValueError(f"No registered loader can handle: {source!r}")


def load(source: str | Path) -> Document:
    """Convenience: dispatch + load in one call."""
    return get_loader(source).load(source)
