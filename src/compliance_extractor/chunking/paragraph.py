"""Paragraph chunker: split on blank lines, pack to target token count.

A paragraph that alone exceeds target_tokens is split with FixedTokenChunker
inside its char window so we never emit a single chunk above budget.
"""
from __future__ import annotations

import re

from compliance_extractor.chunking.base import Chunk, page_span, section_for
from compliance_extractor.chunking.fixed_token import FixedTokenChunker
from compliance_extractor.chunking.tokens import count_tokens
from compliance_extractor.ingest.base import Document, TextBlock

_PARA_RE = re.compile(r"\n\s*\n")


class ParagraphChunker:
    name = "paragraph"

    def __init__(self, target_tokens: int = 1000, sentence_backend: str = "blingfire") -> None:
        self.target_tokens = target_tokens
        self.sentence_backend = sentence_backend

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.full_text
        if not text:
            return []
        spans = self._paragraph_spans(text)
        if not spans:
            return []
        para_tokens = [count_tokens(text[s:e]) for s, e in spans]

        chunks: list[Chunk] = []
        idx = 0
        i = 0
        n = len(spans)
        while i < n:
            if para_tokens[i] > self.target_tokens:
                chunks.extend(self._oversized_split(doc, spans[i], idx_start=idx))
                idx = len(chunks)
                i += 1
                continue
            j = i
            running = 0
            while j < n and running + para_tokens[j] <= self.target_tokens:
                running += para_tokens[j]
                j += 1
            if j == i:
                j = i + 1
                running = para_tokens[i]
            char_start, char_end = spans[i][0], spans[j - 1][1]
            chunk_text = text[char_start:char_end]
            ps, pe = page_span(doc, char_start, char_end)
            section = section_for(doc, char_start, char_end)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{self.name}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    char_start=char_start,
                    char_end=char_end,
                    n_tokens=running,
                    page_start=ps,
                    page_end=pe,
                    section=section,
                    strategy=self.name,
                    metadata={"target_tokens": self.target_tokens},
                )
            )
            idx += 1
            i = j
        return chunks

    @staticmethod
    def _paragraph_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        last = 0
        for m in _PARA_RE.finditer(text):
            seg = text[last:m.start()].strip()
            if seg:
                start = text.find(seg, last)
                spans.append((start, start + len(seg)))
            last = m.end()
        tail = text[last:].strip()
        if tail:
            start = text.find(tail, last)
            spans.append((start, start + len(tail)))
        return spans

    def _oversized_split(self, doc: Document, span: tuple[int, int], idx_start: int) -> list[Chunk]:
        s, e = span
        sub_text = doc.full_text[s:e]
        sub = Document(
            doc_id=doc.doc_id,
            full_text=sub_text,
            blocks=[TextBlock(sub_text, 0, len(sub_text))],
            metadata=doc.metadata,
        )
        ftc = FixedTokenChunker(self.target_tokens, overlap_tokens=0, sentence_backend=self.sentence_backend)
        sub_chunks = ftc.chunk(sub)
        out: list[Chunk] = []
        for k, c in enumerate(sub_chunks):
            cs = s + c.char_start
            ce = s + c.char_end
            ps, pe = page_span(doc, cs, ce)
            section = section_for(doc, cs, ce)
            out.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{self.name}_{idx_start + k:04d}",
                    doc_id=doc.doc_id,
                    text=doc.full_text[cs:ce],
                    char_start=cs,
                    char_end=ce,
                    n_tokens=c.n_tokens,
                    page_start=ps,
                    page_end=pe,
                    section=section,
                    strategy=self.name,
                    metadata={"target_tokens": self.target_tokens, "oversized_paragraph": True},
                )
            )
        return out
