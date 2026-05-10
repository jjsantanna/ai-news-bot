"""Provider-agnostic LLM client interface.

All extractors call into a `LLMClient` via `extract(request) -> response`.
The shape is the same across providers; what differs is whether `citation`
comes back populated by the provider (Anthropic Citations API) or whether
the validator must locate the quote against the source (OpenAI / Ollama /
DeepSeek-Mistral). The decision/quote/source fields are always present so
the downstream validator and IO layer don't have to special-case providers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable


Decision = Literal["complies", "not-complies", "maybe-complies", "no-evidence"]


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class CitedSpan:
    """Anthropic-native citation: char offsets are into the document we sent."""
    doc_id: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    cited_text: str | None = None


@dataclass
class ExtractionRequest:
    control_id: str
    control_title: str
    control_description: str
    documents: list[SourceDocument]
    system_prompt: str
    extra_user_instructions: str | None = None
    cache_documents: bool = True
    max_output_tokens: int = 1024


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class ExtractionResponse:
    decision: Decision
    evidence_quote: str
    evidence_quote_source: str
    rationale: str
    citation: CitedSpan | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    model: str = ""
    provider: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    name: str
    model: str

    def extract(self, request: ExtractionRequest) -> ExtractionResponse: ...
