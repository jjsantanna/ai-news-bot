"""Validation result types and status enums.

The validator answers: did the LLM's `evidence_quote` actually come from
the source document (faithful), or did it paraphrase / fabricate
(hallucinated)? `match_method` reports which rung of the ladder succeeded
so the eval grid can stratify faithfulness by extraction strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ValidationStatus = Literal[
    "validated",       # quote verified against source (exact / normalized / fuzzy)
    "semantic",        # quote matches semantically but not lexically (paraphrase)
    "hallucinated",    # quote does not appear in source at any rung
    "no_evidence",     # LLM said no-evidence; nothing to validate
    "missing_source",  # LLM gave a quote but no doc_id we can locate
]

MatchMethod = Literal[
    "citation",   # Anthropic native citation, already located by the API
    "exact",      # verbatim substring match
    "normalized", # match after whitespace+case+punct normalization
    "fuzzy",      # rapidfuzz partial_ratio >= threshold
    "semantic",   # embedding cosine similarity >= threshold
    "none",       # no rung succeeded
    "skipped",    # validation skipped (no_evidence)
]


@dataclass
class ValidationResult:
    status: ValidationStatus
    match_method: MatchMethod
    similarity_score: float = 0.0
    located_char_start: int | None = None
    located_char_end: int | None = None
    located_page_start: int | None = None
    located_page_end: int | None = None
    located_section: str | None = None
    located_doc_id: str | None = None
    notes: str = ""
    metadata: dict = field(default_factory=dict)
