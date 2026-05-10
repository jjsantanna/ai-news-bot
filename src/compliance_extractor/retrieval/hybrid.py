"""Hybrid retriever: reciprocal rank fusion of BM25 + dense (or any pair)."""
from __future__ import annotations

from typing import Iterable

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.retrieval.base import RetrievalResult, Retriever


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = 60,
        per_retriever_k: int = 20,
    ) -> None:
        if len(retrievers) < 2:
            raise ValueError("HybridRetriever needs at least 2 underlying retrievers")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.per_retriever_k = per_retriever_k

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        contributions: dict[str, dict[str, object]] = {}
        for r in self.retrievers:
            results = r.search(query, k=self.per_retriever_k)
            for res in results:
                cid = res.chunk.chunk_id
                entry = contributions.setdefault(
                    cid,
                    {"chunk": res.chunk, "score": 0.0, "ranks": {}, "scores": {}},
                )
                entry["score"] = float(entry["score"]) + 1.0 / (self.rrf_k + res.rank + 1)
                entry["ranks"][r.name] = res.rank  # type: ignore[index]
                entry["scores"][r.name] = res.score  # type: ignore[index]

        ordered = sorted(contributions.values(), key=lambda e: -float(e["score"]))[:k]
        return [
            RetrievalResult(
                chunk=e["chunk"],  # type: ignore[arg-type]
                score=float(e["score"]),
                rank=rank,
                retriever=self.name,
                metadata={
                    "rrf_k": self.rrf_k,
                    "components": [r.name for r in self.retrievers],
                    "component_ranks": e["ranks"],
                    "component_scores": e["scores"],
                },
            )
            for rank, e in enumerate(ordered)
        ]
