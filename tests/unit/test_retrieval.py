"""Tests for retrievers.

BM25 runs against a small in-memory corpus end-to-end. Dense, hybrid, and
reranker tests use injected fake encoders / score functions so the suite
doesn't require sentence-transformers model downloads.
"""
from __future__ import annotations

import hashlib

import pytest

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.retrieval.bm25 import BM25Retriever
from compliance_extractor.retrieval.dense import DenseRetriever
from compliance_extractor.retrieval.hybrid import HybridRetriever
from compliance_extractor.retrieval.registry import build_retriever, list_strategies
from compliance_extractor.retrieval.reranker import RerankRetriever


def _mk_chunk(idx: int, text: str, doc_id: str = "d") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#c_{idx:04d}",
        doc_id=doc_id,
        text=text,
        char_start=0,
        char_end=len(text),
        n_tokens=max(1, len(text) // 4),
        strategy="test",
    )


CORPUS_TEXTS = [
    "Multi-factor authentication is required for all administrators on production systems.",
    "Backups are taken nightly and encrypted at rest using AES-256.",
    "Incident response runbooks are maintained in the security wiki.",
    "Access reviews are performed quarterly for privileged users.",
    "Password complexity is enforced with a minimum of twelve characters.",
]


@pytest.fixture
def corpus() -> list[Chunk]:
    return [_mk_chunk(i, t) for i, t in enumerate(CORPUS_TEXTS)]


@pytest.mark.fast
def test_bm25_ranks_topical_chunk_first(corpus) -> None:
    pytest.importorskip("rank_bm25")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    results = bm25.search("multi-factor authentication for administrators", k=3)
    assert results
    assert results[0].chunk.text == CORPUS_TEXTS[0]
    assert results[0].rank == 0
    assert results[0].retriever == "bm25"
    assert results[0].score > 0


@pytest.mark.fast
def test_bm25_no_match_returns_empty(corpus) -> None:
    pytest.importorskip("rank_bm25")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    results = bm25.search("zzz_xyz_no_such_term", k=3)
    assert results == []


def _hash_encode(texts: list[str], dim: int = 16):
    """Deterministic, dependency-free fake encoder: hash-based bag-of-words."""
    import numpy as np

    vecs = []
    for t in texts:
        v = [0.0] * dim
        for tok in t.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        arr = np.asarray(v, dtype="float32")
        n = float(np.linalg.norm(arr) or 1.0)
        vecs.append(arr / n)
    return np.stack(vecs)


@pytest.mark.fast
def test_dense_with_fake_encoder(corpus, tmp_path) -> None:
    pytest.importorskip("chromadb")
    pytest.importorskip("numpy")
    dr = DenseRetriever(encoder=_hash_encode, persist_dir=str(tmp_path / "chroma"), collection_name="test_dense")
    dr.index(corpus)
    results = dr.search("authentication administrators", k=3)
    assert results
    assert all(r.retriever == "dense" for r in results)
    assert results[0].rank == 0


@pytest.mark.fast
def test_hybrid_rrf_combines_two_retrievers(corpus, tmp_path) -> None:
    pytest.importorskip("rank_bm25")
    pytest.importorskip("chromadb")
    bm25 = BM25Retriever()
    dense = DenseRetriever(encoder=_hash_encode, persist_dir=str(tmp_path / "chroma"), collection_name="test_hybrid")
    hybrid = HybridRetriever([bm25, dense], rrf_k=60, per_retriever_k=5)
    hybrid.index(corpus)
    results = hybrid.search("authentication administrators", k=3)
    assert results
    assert results[0].retriever == "hybrid"
    component_meta = results[0].metadata["component_ranks"]
    assert "bm25" in component_meta or "dense" in component_meta


@pytest.mark.fast
def test_rerank_reorders_by_score_fn(corpus) -> None:
    pytest.importorskip("rank_bm25")
    bm25 = BM25Retriever()

    def fake_score(pairs):
        return [1.0 if "incident" in t.lower() else 0.0 for _, t in pairs]

    rr = RerankRetriever(base=bm25, score_fn=fake_score, first_stage_k=5)
    rr.index(corpus)
    results = rr.search("administrators backups incident reviews password", k=3)
    assert results
    assert "incident" in results[0].chunk.text.lower()
    assert results[0].retriever.endswith("+rerank")


@pytest.mark.fast
def test_registry_dispatch_and_listing() -> None:
    pytest.importorskip("rank_bm25")
    assert set(list_strategies()) == {"bm25", "dense", "hybrid", "rerank"}
    bm = build_retriever("bm25")
    assert bm.name == "bm25"
    with pytest.raises(ValueError, match="Unknown retrieval"):
        build_retriever("nope")


@pytest.mark.fast
def test_hybrid_requires_two_underlying_retrievers() -> None:
    pytest.importorskip("rank_bm25")
    with pytest.raises(ValueError, match="at least 2"):
        HybridRetriever([BM25Retriever()])
