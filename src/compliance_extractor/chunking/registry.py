"""Chunker registry: build a Chunker from a variant config name + params."""
from __future__ import annotations

from typing import Any

from compliance_extractor.chunking.base import Chunker
from compliance_extractor.chunking.fixed_token import FixedTokenChunker
from compliance_extractor.chunking.paragraph import ParagraphChunker
from compliance_extractor.chunking.section import SectionChunker
from compliance_extractor.chunking.semantic import SemanticChunker

_BUILDERS: dict[str, Any] = {
    "fixed_token": FixedTokenChunker,
    "paragraph": ParagraphChunker,
    "section": SectionChunker,
    "semantic": SemanticChunker,
}


def build_chunker(strategy: str, **kwargs: Any) -> Chunker:
    if strategy not in _BUILDERS:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Valid: {sorted(_BUILDERS)}")
    return _BUILDERS[strategy](**kwargs)


def list_strategies() -> list[str]:
    return sorted(_BUILDERS)
