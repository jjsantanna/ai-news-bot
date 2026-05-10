"""RAG extractor: chunk -> retrieve top-k -> pass chunks as docs to LLM.

Each retrieved chunk becomes a SourceDocument with `doc_id` rewritten as
`{original_doc_id}#chunk_{i}` so the validator can both (a) match the
quote against the chunk text we actually showed the LLM and (b) recover
the original document via the chunk's char offsets at locator time.
"""
from __future__ import annotations

from compliance_extractor.extractors.base import ExtractorContext
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    SourceDocument,
)


class RAGExtractor:
    name = "rag"

    def extract(self, ctx: ExtractorContext) -> ExtractionResponse:
        if ctx.chunker is None or ctx.retriever is None:
            raise ValueError("RAGExtractor requires both chunker and retriever in context")
        all_chunks = []
        for doc in ctx.documents:
            all_chunks.extend(ctx.chunker.chunk(doc))
        ctx.retriever.index(all_chunks)
        query = _build_query(ctx)
        results = ctx.retriever.search(query, k=ctx.top_k)

        source_docs = [
            SourceDocument(
                doc_id=r.chunk.chunk_id,
                text=r.chunk.text,
                title=f"{r.chunk.doc_id} (chunk {r.rank})",
            )
            for r in results
        ]
        if not source_docs:
            for doc in ctx.documents:
                source_docs.append(SourceDocument(doc_id=doc.doc_id, text=doc.full_text, title=doc.doc_id))

        request = ExtractionRequest(
            control_id=ctx.control.control_id,
            control_title=ctx.control.title,
            control_description=ctx.control.description,
            documents=source_docs,
            system_prompt="",
            max_output_tokens=ctx.max_output_tokens,
        )
        response = ctx.llm.extract(request)
        response.raw = {**response.raw, "extractor": self.name, "n_retrieved": len(results)}
        return response


def _build_query(ctx: ExtractorContext) -> str:
    return f"{ctx.control.control_id}: {ctx.control.title}\n{ctx.control.description}"
