"""Cross-encoder reranker that decorates any base retriever.

The base retriever's top `first_stage_k` results are scored with a
cross-encoder and re-ranked. `score_fn` is injected for testability; the
default loads `BAAI/bge-reranker-base` via sentence-transformers'
CrossEncoder lazily.
"""
from __future__ import annotations

from typing import Callable, Iterable

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.retrieval.base import RetrievalResult, Retriever


def default_cross_encoder(model_name: str = "BAAI/bge-reranker-base") -> Callable[[list[tuple[str, str]]], list[float]]:
    def _score(pairs: list[tuple[str, str]]) -> list[float]:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        if not hasattr(_score, "_model"):
            _score._model = CrossEncoder(model_name)  # type: ignore[attr-defined]
        return [float(s) for s in _score._model.predict(pairs)]  # type: ignore[attr-defined]

    return _score


class RerankRetriever:
    name = "rerank"

    def __init__(
        self,
        base: Retriever,
        score_fn: Callable[[list[tuple[str, str]]], list[float]] | None = None,
        model_name: str = "BAAI/bge-reranker-base",
        first_stage_k: int = 20,
    ) -> None:
        self.base = base
        self.score_fn = score_fn if score_fn is not None else default_cross_encoder(model_name)
        self.model_name = model_name
        self.first_stage_k = first_stage_k

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.base.index(chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        first = self.base.search(query, k=self.first_stage_k)
        if not first:
            return []
        pairs = [(query, r.chunk.text) for r in first]
        scores = self.score_fn(pairs)
        scored = sorted(zip(first, scores), key=lambda t: -t[1])[:k]
        return [
            RetrievalResult(
                chunk=res.chunk,
                score=float(score),
                rank=rank,
                retriever=f"{self.base.name}+rerank",
                metadata={
                    "base_retriever": self.base.name,
                    "base_rank": res.rank,
                    "base_score": res.score,
                    "rerank_model": self.model_name,
                },
            )
            for rank, (res, score) in enumerate(scored)
        ]
