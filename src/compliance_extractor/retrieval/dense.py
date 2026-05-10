"""Dense retriever: pluggable encoder + ChromaDB persistent collection.

The `encoder` is a callable `list[str] -> np.ndarray` so tests can inject a
deterministic fake; the default uses sentence-transformers
(`all-MiniLM-L6-v2`). Indexing is idempotent per `collection_name`: if the
collection already has the corpus indexed, re-indexing is a no-op.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from compliance_extractor.chunking.base import Chunk
from compliance_extractor.retrieval.base import RetrievalResult


def default_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[list[str]], "object"]:
    def _encode(texts: list[str]):
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        if not hasattr(_encode, "_model"):
            _encode._model = SentenceTransformer(model_name)  # type: ignore[attr-defined]
        return _encode._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)  # type: ignore[attr-defined]

    return _encode


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        encoder: Callable[[list[str]], "object"] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "default",
        persist_dir: str | Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self.encoder = encoder if encoder is not None else default_encoder(model_name)
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_dir = str(persist_dir) if persist_dir else None
        self.batch_size = batch_size
        self._chunks: dict[str, Chunk] = {}
        self._collection = None
        self._client = None

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb  # type: ignore[import-not-found]

        if self.persist_dir:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def index(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        self._chunks = {c.chunk_id: c for c in chunks}
        coll = self._ensure_collection()

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]
            embeddings = self.encoder(texts)
            embeddings_list = [list(map(float, row)) for row in embeddings]
            coll.upsert(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings_list,
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "n_tokens": c.n_tokens,
                        "page_start": c.page_start if c.page_start is not None else -1,
                        "page_end": c.page_end if c.page_end is not None else -1,
                        "section": c.section or "",
                        "strategy": c.strategy,
                    }
                    for c in batch
                ],
            )

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if not self._chunks:
            return []
        coll = self._ensure_collection()
        q_emb = self.encoder([query])
        q_list = [list(map(float, row)) for row in q_emb]
        res = coll.query(query_embeddings=q_list, n_results=k)
        ids = res["ids"][0]
        distances = res["distances"][0]
        out: list[RetrievalResult] = []
        for rank, (chunk_id, dist) in enumerate(zip(ids, distances)):
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            similarity = max(0.0, 1.0 - float(dist))
            out.append(
                RetrievalResult(
                    chunk=chunk,
                    score=similarity,
                    rank=rank,
                    retriever=self.name,
                    metadata={"model": self.model_name, "distance": float(dist)},
                )
            )
        return out
