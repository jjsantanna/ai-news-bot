"""Retriever registry: build a Retriever from a strategy name + params."""
from __future__ import annotations

from typing import Any

from compliance_extractor.retrieval.base import Retriever
from compliance_extractor.retrieval.bm25 import BM25Retriever
from compliance_extractor.retrieval.dense import DenseRetriever
from compliance_extractor.retrieval.hybrid import HybridRetriever
from compliance_extractor.retrieval.reranker import RerankRetriever


def build_retriever(strategy: str, **kwargs: Any) -> Retriever:
    if strategy == "bm25":
        return BM25Retriever(**kwargs)
    if strategy == "dense":
        return DenseRetriever(**kwargs)
    if strategy == "hybrid":
        bm25_kwargs = kwargs.pop("bm25", {})
        dense_kwargs = kwargs.pop("dense", {})
        retrievers = [BM25Retriever(**bm25_kwargs), DenseRetriever(**dense_kwargs)]
        return HybridRetriever(retrievers=retrievers, **kwargs)
    if strategy == "rerank":
        base_strategy = kwargs.pop("base", "hybrid")
        base_kwargs = kwargs.pop("base_kwargs", {})
        base = build_retriever(base_strategy, **base_kwargs)
        return RerankRetriever(base=base, **kwargs)
    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")


def list_strategies() -> list[str]:
    return ["bm25", "dense", "hybrid", "rerank"]
