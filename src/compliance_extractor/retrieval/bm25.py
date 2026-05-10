"""BM25 retriever using rank_bm25 (pure Python, deterministic, no model load)."""
from __future__ import annotations

import re
from typing import Iterable

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.retrieval.base import RetrievalResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def index(self, chunks: Iterable[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if not self._chunks or self._bm25 is None:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(self._chunks)), key=lambda i: -scores[i])[:k]
        return [
            RetrievalResult(
                chunk=self._chunks[i],
                score=float(scores[i]),
                rank=rank,
                retriever=self.name,
                metadata={"k1": self.k1, "b": self.b},
            )
            for rank, i in enumerate(order)
            if scores[i] > 0
        ]
