"""Long-context extractor: pass full corpus to the LLM, no retrieval.

Best paired with claude-opus-4-7's 1M context. Tests RQ1 directly:
does the LLM extract better when given the full corpus vs RAG slices?
"""
from __future__ import annotations

from compliance_extractor.extractors.base import ExtractorContext
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    SourceDocument,
)


class LongContextExtractor:
    name = "long_context"

    def extract(self, ctx: ExtractorContext) -> ExtractionResponse:
        source_docs = [
            SourceDocument(doc_id=doc.doc_id, text=doc.full_text, title=doc.doc_id)
            for doc in ctx.documents
        ]
        request = ExtractionRequest(
            control_id=ctx.control.control_id,
            control_title=ctx.control.title,
            control_description=ctx.control.description,
            documents=source_docs,
            system_prompt="",
            max_output_tokens=ctx.max_output_tokens,
        )
        response = ctx.llm.extract(request)
        response.raw = {**response.raw, "extractor": self.name, "n_documents": len(source_docs)}
        return response
