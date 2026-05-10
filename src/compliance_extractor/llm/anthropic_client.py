"""Anthropic client with native Citations API + prompt caching.

Documents are sent as content blocks of type `document` with `citations:
{enabled: True}`. The model returns interleaved text and citation blocks;
we extract the JSON decision from the text and the per-quote span from
the citation blocks (when present). Prompt caching is applied to the last
document block so repeated controls against the same corpus reuse the
cached document context.

References:
  - https://docs.anthropic.com/en/docs/build-with-claude/citations
  - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from compliance_extractor.llm.base import (
    CitedSpan,
    ExtractionRequest,
    ExtractionResponse,
    SourceDocument,
    TokenUsage,
)
from compliance_extractor.llm.cache import get_cached, make_cache_key, set_cached
from compliance_extractor.llm.parsing import parse_extraction_payload
from compliance_extractor.llm.prompts import SYSTEM_PROMPT, render_user_prompt

DEFAULT_MODEL = "claude-opus-4-7"


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Any | None = None,
        use_cache: bool = True,
        request_extras: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._client = client
        self.use_cache = use_cache
        self.request_extras = request_extras or {}

    def _get_client(self):
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else anthropic.Anthropic()
        return self._client

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        user_prompt = render_user_prompt(request)
        cache_key = make_cache_key(
            provider=self.name,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            documents=[(d.doc_id, d.text) for d in request.documents],
            extra={"max_tokens": request.max_output_tokens},
        )
        if self.use_cache:
            hit = get_cached(cache_key)
            if hit is not None:
                return _from_cache_dict(hit)

        client = self._get_client()
        document_blocks = _build_document_blocks(request.documents, cache_documents=request.cache_documents)
        messages = [
            {
                "role": "user",
                "content": [
                    *document_blocks,
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        t0 = time.perf_counter()
        response = client.messages.create(
            model=self.model,
            max_tokens=request.max_output_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
            **self.request_extras,
        )
        latency = time.perf_counter() - t0

        text_parts: list[str] = []
        first_citation: CitedSpan | None = None
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
                if first_citation is None:
                    citations = getattr(block, "citations", None) or []
                    for c in citations:
                        first_citation = _citation_to_span(c, request.documents)
                        if first_citation is not None:
                            break

        text = "".join(text_parts).strip()
        decision, quote, source_id, rationale = parse_extraction_payload(text)
        usage = _extract_usage(response)

        result = ExtractionResponse(
            decision=decision,
            evidence_quote=quote,
            evidence_quote_source=source_id,
            rationale=rationale,
            citation=first_citation,
            usage=usage,
            latency_seconds=latency,
            model=self.model,
            provider=self.name,
            raw={"response_text": text},
        )
        if self.use_cache:
            set_cached(cache_key, _to_cache_dict(result))
        return result


def _build_document_blocks(documents: list[SourceDocument], cache_documents: bool) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for i, doc in enumerate(documents):
        block: dict[str, Any] = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": doc.text},
            "title": doc.title or doc.doc_id,
            "context": f"doc_id={doc.doc_id}",
            "citations": {"enabled": True},
        }
        if cache_documents and i == len(documents) - 1:
            block["cache_control"] = {"type": "ephemeral"}
        blocks.append(block)
    return blocks


def _citation_to_span(citation: Any, documents: list[SourceDocument]) -> CitedSpan | None:
    ctype = getattr(citation, "type", None)
    if ctype not in {"char_location", "page_location", "content_block_location"}:
        return None
    doc_index = getattr(citation, "document_index", 0) or 0
    doc_id = documents[doc_index].doc_id if 0 <= doc_index < len(documents) else ""
    cited_text = getattr(citation, "cited_text", None)
    if ctype == "char_location":
        return CitedSpan(
            doc_id=doc_id,
            char_start=int(getattr(citation, "start_char_index", 0)),
            char_end=int(getattr(citation, "end_char_index", 0)),
            cited_text=cited_text,
        )
    if ctype == "page_location":
        return CitedSpan(
            doc_id=doc_id,
            char_start=0,
            char_end=0,
            page_start=int(getattr(citation, "start_page_number", 0)),
            page_end=int(getattr(citation, "end_page_number", 0)),
            cited_text=cited_text,
        )
    return CitedSpan(
        doc_id=doc_id,
        char_start=int(getattr(citation, "start_block_index", 0)),
        char_end=int(getattr(citation, "end_block_index", 0)),
        cited_text=cited_text,
    )


def _extract_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


def _to_cache_dict(resp: ExtractionResponse) -> dict[str, Any]:
    d = asdict(resp)
    if resp.citation is not None:
        d["citation"] = asdict(resp.citation)
    return d


def _from_cache_dict(d: dict[str, Any]) -> ExtractionResponse:
    citation = None
    if d.get("citation"):
        citation = CitedSpan(**d["citation"])
    usage = TokenUsage(**(d.get("usage") or {}))
    return ExtractionResponse(
        decision=d["decision"],
        evidence_quote=d["evidence_quote"],
        evidence_quote_source=d["evidence_quote_source"],
        rationale=d.get("rationale", ""),
        citation=citation,
        usage=usage,
        latency_seconds=float(d.get("latency_seconds", 0.0)),
        model=d.get("model", ""),
        provider=d.get("provider", ""),
        raw=d.get("raw", {}),
    )
