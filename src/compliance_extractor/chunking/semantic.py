"""Semantic chunker: cut at sentence-similarity drops.

Uses sentence-transformers (default `all-MiniLM-L6-v2`) to embed sentences and
cuts at adjacency cosine-similarity gaps falling below a percentile threshold.
The boundary set is then packed into chunks under target_tokens. Heavy deps
are imported lazily so the module is importable without sentence-transformers
installed; `chunk()` raises a clear error on first call if it's missing.
"""
from __future__ import annotations

from compliance_extractor.chunking.base import Chunk, page_span, section_for
from compliance_extractor.chunking.sentence_splitter import split_sentences
from compliance_extractor.chunking.tokens import count_tokens
from compliance_extractor.ingest.base import Document


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        target_tokens: int = 1000,
        model: str = "all-MiniLM-L6-v2",
        breakpoint_percentile: float = 90.0,
        sentence_backend: str = "blingfire",
    ) -> None:
        self.target_tokens = target_tokens
        self.model_name = model
        self.breakpoint_percentile = breakpoint_percentile
        self.sentence_backend = sentence_backend
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "SemanticChunker requires `sentence-transformers`; "
                    "install it or pick a different chunking strategy."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.full_text, backend=self.sentence_backend)
        if not sentences:
            return []
        if len(sentences) == 1:
            return self._single_chunk(doc, sentences)

        import numpy as np

        model = self._ensure_model()
        texts = [s for s, _, _ in sentences]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims = (embeddings[:-1] * embeddings[1:]).sum(axis=1)
        gaps = 1.0 - sims

        threshold = float(np.percentile(gaps, self.breakpoint_percentile)) if len(gaps) else 1.0
        cuts = {i + 1 for i, g in enumerate(gaps) if g >= threshold}

        sent_tokens = [count_tokens(t) for t in texts]

        chunks: list[Chunk] = []
        idx = 0
        i = 0
        n = len(sentences)
        while i < n:
            j = i
            running = 0
            while j < n and running + sent_tokens[j] <= self.target_tokens:
                running += sent_tokens[j]
                j += 1
                if j in cuts:
                    break
            if j == i:
                j = i + 1
                running = sent_tokens[i]
            chunks.append(self._make(doc, sentences, i, j, running, idx))
            idx += 1
            i = j
        return chunks

    def _single_chunk(self, doc: Document, sentences: list[tuple[str, int, int]]) -> list[Chunk]:
        s, cs, ce = sentences[0]
        return [self._make(doc, sentences, 0, 1, count_tokens(s), 0)]

    def _make(self, doc, sentences, i, j, running, idx) -> Chunk:
        char_start = sentences[i][1]
        char_end = sentences[j - 1][2]
        text = doc.full_text[char_start:char_end]
        ps, pe = page_span(doc, char_start, char_end)
        section = section_for(doc, char_start, char_end)
        return Chunk(
            chunk_id=f"{doc.doc_id}#{self.name}_{idx:04d}",
            doc_id=doc.doc_id,
            text=text,
            char_start=char_start,
            char_end=char_end,
            n_tokens=running,
            page_start=ps,
            page_end=pe,
            section=section,
            strategy=self.name,
            metadata={
                "target_tokens": self.target_tokens,
                "model": self.model_name,
                "breakpoint_percentile": self.breakpoint_percentile,
            },
        )
