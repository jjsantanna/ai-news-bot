"""Section chunker: heading-aware grouping with token-budget subdivision.

Groups adjacent TextBlocks that share the same `section` path; if the
resulting region exceeds target_tokens, it's subdivided via the fixed-token
chunker (sentence-packing). Falls back to paragraph chunking if the source
Document carries no section metadata.
"""
from __future__ import annotations

from itertools import groupby

from compliance_extractor.chunking.base import Chunk, page_span
from compliance_extractor.chunking.fixed_token import FixedTokenChunker
from compliance_extractor.chunking.paragraph import ParagraphChunker
from compliance_extractor.chunking.tokens import count_tokens
from compliance_extractor.ingest.base import Document, TextBlock


class SectionChunker:
    name = "section"

    def __init__(self, target_tokens: int = 1000, sentence_backend: str = "blingfire") -> None:
        self.target_tokens = target_tokens
        self.sentence_backend = sentence_backend

    def chunk(self, doc: Document) -> list[Chunk]:
        if not any(b.section for b in doc.blocks):
            return ParagraphChunker(self.target_tokens, self.sentence_backend).chunk(doc)

        chunks: list[Chunk] = []
        idx = 0
        for section, group_iter in groupby(doc.blocks, key=lambda b: b.section):
            group = list(group_iter)
            if not group:
                continue
            char_start = group[0].char_start
            char_end = group[-1].char_end
            text = doc.full_text[char_start:char_end]
            ntok = count_tokens(text)
            if ntok <= self.target_tokens:
                ps, pe = page_span(doc, char_start, char_end)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{self.name}_{idx:04d}",
                        doc_id=doc.doc_id,
                        text=text,
                        char_start=char_start,
                        char_end=char_end,
                        n_tokens=ntok,
                        page_start=ps,
                        page_end=pe,
                        section=section,
                        strategy=self.name,
                        metadata={"target_tokens": self.target_tokens},
                    )
                )
                idx += 1
            else:
                sub = Document(
                    doc_id=doc.doc_id,
                    full_text=text,
                    blocks=[TextBlock(text, 0, len(text), section=section)],
                    metadata=doc.metadata,
                )
                ftc = FixedTokenChunker(self.target_tokens, overlap_tokens=0, sentence_backend=self.sentence_backend)
                for c in ftc.chunk(sub):
                    cs = char_start + c.char_start
                    ce = char_start + c.char_end
                    ps, pe = page_span(doc, cs, ce)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}#{self.name}_{idx:04d}",
                            doc_id=doc.doc_id,
                            text=doc.full_text[cs:ce],
                            char_start=cs,
                            char_end=ce,
                            n_tokens=c.n_tokens,
                            page_start=ps,
                            page_end=pe,
                            section=section,
                            strategy=self.name,
                            metadata={"target_tokens": self.target_tokens, "subdivided_section": True},
                        )
                    )
                    idx += 1
        return chunks
