"""Retriever protocol and shared result types.

Retrievers operate on an iterable of `Chunk`s and return ranked
`RetrievalResult`s. The result carries the chunk reference, a normalized
score, and provenance fields the eval pipeline copies into the candidate
DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from compliance_extractor.chunking.base import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, chunks: Iterable[Chunk]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]: ...
