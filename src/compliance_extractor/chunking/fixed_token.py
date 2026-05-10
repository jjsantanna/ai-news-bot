"""Fixed-token chunker: sentence-aware token windows with configurable overlap.

Sentences are tokenized with tiktoken and packed greedily until adding the
next sentence would exceed `target_tokens`. Overlap is realized by re-emitting
the trailing `overlap_tokens` worth of sentences at the start of the next
chunk. Char offsets always span the first→last sentence in the chunk.
"""
from __future__ import annotations

from compliance_extractor.chunking.base import Chunk, page_span, section_for
from compliance_extractor.chunking.sentence_splitter import split_sentences
from compliance_extractor.chunking.tokens import count_tokens
from compliance_extractor.ingest.base import Document


class FixedTokenChunker:
    name = "fixed_token"

    def __init__(
        self,
        target_tokens: int = 1000,
        overlap_tokens: int = 100,
        sentence_backend: str = "blingfire",
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("0 <= overlap_tokens < target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.sentence_backend = sentence_backend

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = split_sentences(doc.full_text, backend=self.sentence_backend)
        if not sentences:
            return []
        sent_tokens = [count_tokens(s) for s, _, _ in sentences]

        chunks: list[Chunk] = []
        i = 0
        n = len(sentences)
        idx = 0
        while i < n:
            j = i
            running = 0
            while j < n and running + sent_tokens[j] <= self.target_tokens:
                running += sent_tokens[j]
                j += 1
            if j == i:
                j = i + 1
                running = sent_tokens[i]

            text_start = sentences[i][1]
            text_end = sentences[j - 1][2]
            text = doc.full_text[text_start:text_end]
            ps, pe = page_span(doc, text_start, text_end)
            section = section_for(doc, text_start, text_end)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{self.name}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    char_start=text_start,
                    char_end=text_end,
                    n_tokens=running,
                    page_start=ps,
                    page_end=pe,
                    section=section,
                    strategy=self.name,
                    metadata={"target_tokens": self.target_tokens, "overlap_tokens": self.overlap_tokens},
                )
            )
            idx += 1

            if j >= n:
                break
            if self.overlap_tokens == 0:
                i = j
                continue
            back = 0
            k = j
            while k > i and back < self.overlap_tokens:
                k -= 1
                back += sent_tokens[k]
            i = max(k, i + 1)
        return chunks
