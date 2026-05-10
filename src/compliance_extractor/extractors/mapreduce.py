"""Map-reduce extractor: per-chunk LLM call, then pick the best.

Map: send each chunk to the LLM separately and collect per-chunk decisions
(complies / not-complies / maybe-complies / no-evidence) with quotes.
Reduce: pick the strongest signal per the decision priority and return
its quote. This is the most expensive strategy (one LLM call per chunk)
but gives the model maximum local context per call.
"""
from __future__ import annotations

from compliance_extractor.extractors.base import ExtractorContext
from compliance_extractor.llm.base import (
    ExtractionRequest,
    ExtractionResponse,
    SourceDocument,
    TokenUsage,
)


_DECISION_RANK = {
    "complies": 4,
    "not-complies": 3,
    "maybe-complies": 2,
    "no-evidence": 1,
}


class MapReduceExtractor:
    name = "mapreduce"

    def extract(self, ctx: ExtractorContext) -> ExtractionResponse:
        if ctx.chunker is None:
            raise ValueError("MapReduceExtractor requires a chunker in context")
        all_chunks = []
        for doc in ctx.documents:
            all_chunks.extend(ctx.chunker.chunk(doc))
        if not all_chunks:
            return _no_evidence_response(ctx, reason="no chunks")

        per_chunk_responses: list[ExtractionResponse] = []
        for chunk in all_chunks:
            request = ExtractionRequest(
                control_id=ctx.control.control_id,
                control_title=ctx.control.title,
                control_description=ctx.control.description,
                documents=[SourceDocument(doc_id=chunk.chunk_id, text=chunk.text, title=chunk.doc_id)],
                system_prompt="",
                max_output_tokens=ctx.max_output_tokens,
            )
            per_chunk_responses.append(ctx.llm.extract(request))

        best = max(per_chunk_responses, key=lambda r: _DECISION_RANK.get(r.decision, 0))
        merged_usage = TokenUsage(
            input_tokens=sum(r.usage.input_tokens for r in per_chunk_responses),
            output_tokens=sum(r.usage.output_tokens for r in per_chunk_responses),
            cache_read_input_tokens=sum(r.usage.cache_read_input_tokens for r in per_chunk_responses),
            cache_creation_input_tokens=sum(r.usage.cache_creation_input_tokens for r in per_chunk_responses),
        )
        best.usage = merged_usage
        best.latency_seconds = sum(r.latency_seconds for r in per_chunk_responses)
        best.raw = {**best.raw, "extractor": self.name, "n_chunks": len(all_chunks)}
        return best


def _no_evidence_response(ctx: ExtractorContext, reason: str) -> ExtractionResponse:
    return ExtractionResponse(
        decision="no-evidence",
        evidence_quote="",
        evidence_quote_source="",
        rationale=f"map-reduce produced no candidates ({reason})",
        provider=ctx.llm.name,
        model=ctx.llm.model,
        raw={"extractor": "mapreduce"},
    )
