"""Extractor registry: build by strategy name."""
from __future__ import annotations

from compliance_extractor.extractors.base import Extractor
from compliance_extractor.extractors.citations_rag import CitationsRAGExtractor
from compliance_extractor.extractors.long_context import LongContextExtractor
from compliance_extractor.extractors.mapreduce import MapReduceExtractor
from compliance_extractor.extractors.rag import RAGExtractor

_BUILDERS = {
    "rag": RAGExtractor,
    "long_context": LongContextExtractor,
    "mapreduce": MapReduceExtractor,
    "citations_rag": CitationsRAGExtractor,
}


def build_extractor(strategy: str) -> Extractor:
    if strategy not in _BUILDERS:
        raise ValueError(f"Unknown extractor strategy: {strategy!r}. Valid: {sorted(_BUILDERS)}")
    return _BUILDERS[strategy]()


def list_strategies() -> list[str]:
    return sorted(_BUILDERS)
