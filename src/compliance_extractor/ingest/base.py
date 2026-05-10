"""Document loader protocol + canonical Document type.

Every loader produces the same `Document` shape regardless of modality. The
`full_text` is the canonical source against which the validator's exact /
normalized / fuzzy / semantic rungs run; `blocks` carry the page / section
metadata used by the locator to backfill page+line numbers in the evidence
xlsx. Char offsets in `blocks` are slices into `full_text`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TextBlock:
    text: str
    char_start: int
    char_end: int
    page: int | None = None
    section: str | None = None


@dataclass
class Document:
    doc_id: str
    full_text: str
    blocks: list[TextBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def slice(self, start: int, end: int) -> str:
        return self.full_text[start:end]


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol every loader satisfies."""

    name: str

    def can_handle(self, source: str | Path) -> bool: ...

    def load(self, source: str | Path) -> Document: ...
