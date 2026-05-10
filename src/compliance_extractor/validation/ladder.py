"""Faithfulness validator: exact → normalized → fuzzy → semantic.

Given an evidence_quote and a corpus of documents, the ladder reports the
first rung that succeeds. `validated` means the quote demonstrably came
from the source; `semantic` means the meaning is in the source but the
quote was paraphrased (a soft hallucination); `hallucinated` means the
quote is nowhere in the source.

The ladder accepts an optional Anthropic CitedSpan; if present, we treat
the citation as a `citation` match (verifying the cited_text against
the document) and skip the rest of the ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from compliance_extractor.ingest.base import Document
from compliance_extractor.llm.base import CitedSpan
from compliance_extractor.validation.locator import locate
from compliance_extractor.validation.normalize import normalize_for_match
from compliance_extractor.validation.types import ValidationResult


@dataclass
class ValidatorConfig:
    fuzzy_threshold: float = 85.0
    semantic_threshold: float = 0.75
    enable_semantic: bool = False
    semantic_score_fn: Callable[[str, str], float] | None = None


class Validator:
    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()

    def validate(
        self,
        evidence_quote: str,
        evidence_quote_source: str,
        documents: Iterable[Document],
        citation: CitedSpan | None = None,
    ) -> ValidationResult:
        documents = list(documents)
        if not evidence_quote:
            return ValidationResult(status="no_evidence", match_method="skipped")

        if citation is not None:
            cit_result = self._from_citation(citation, documents)
            if cit_result is not None:
                return cit_result

        target_doc = self._pick_doc(evidence_quote_source, documents)
        if target_doc is None:
            return ValidationResult(
                status="missing_source",
                match_method="none",
                notes=f"Source doc_id {evidence_quote_source!r} not found in corpus",
            )

        exact = self._exact(evidence_quote, target_doc)
        if exact is not None:
            return exact

        normalized = self._normalized(evidence_quote, target_doc)
        if normalized is not None:
            return normalized

        fuzzy = self._fuzzy(evidence_quote, target_doc)
        if fuzzy is not None:
            return fuzzy

        if self.config.enable_semantic:
            semantic = self._semantic(evidence_quote, target_doc)
            if semantic is not None:
                return semantic

        return ValidationResult(
            status="hallucinated",
            match_method="none",
            located_doc_id=target_doc.doc_id,
            notes="Quote not found at any rung of the validation ladder",
        )

    @staticmethod
    def _pick_doc(source_id: str, documents: list[Document]) -> Document | None:
        if source_id:
            for d in documents:
                if d.doc_id == source_id:
                    return d
            return None
        if len(documents) == 1:
            return documents[0]
        return None

    def _from_citation(self, citation: CitedSpan, documents: list[Document]) -> ValidationResult | None:
        doc = next((d for d in documents if d.doc_id == citation.doc_id), None)
        if doc is None:
            return None
        cs, ce = citation.char_start, citation.char_end
        if 0 <= cs < ce <= len(doc.full_text):
            ps, pe, section = locate(doc, cs, ce)
            return ValidationResult(
                status="validated",
                match_method="citation",
                similarity_score=1.0,
                located_char_start=cs,
                located_char_end=ce,
                located_page_start=ps,
                located_page_end=pe,
                located_section=section,
                located_doc_id=doc.doc_id,
                notes="Anthropic native citation",
            )
        return None

    def _exact(self, quote: str, doc: Document) -> ValidationResult | None:
        idx = doc.full_text.find(quote)
        if idx < 0:
            return None
        cs, ce = idx, idx + len(quote)
        ps, pe, section = locate(doc, cs, ce)
        return ValidationResult(
            status="validated",
            match_method="exact",
            similarity_score=1.0,
            located_char_start=cs,
            located_char_end=ce,
            located_page_start=ps,
            located_page_end=pe,
            located_section=section,
            located_doc_id=doc.doc_id,
        )

    def _normalized(self, quote: str, doc: Document) -> ValidationResult | None:
        n_quote = normalize_for_match(quote)
        if not n_quote:
            return None
        n_text = normalize_for_match(doc.full_text)
        if n_quote not in n_text:
            return None
        cs, ce = self._locate_normalized(quote, doc.full_text)
        if cs is None or ce is None:
            return None
        ps, pe, section = locate(doc, cs, ce)
        return ValidationResult(
            status="validated",
            match_method="normalized",
            similarity_score=1.0,
            located_char_start=cs,
            located_char_end=ce,
            located_page_start=ps,
            located_page_end=pe,
            located_section=section,
            located_doc_id=doc.doc_id,
            notes="Whitespace/punctuation/case-insensitive match",
        )

    @staticmethod
    def _locate_normalized(quote: str, full_text: str) -> tuple[int | None, int | None]:
        n_quote = normalize_for_match(quote)
        words = [w for w in n_quote.split() if w]
        if not words:
            return None, None
        first, last = words[0], words[-1]
        text_lower = full_text.lower()
        start_idx = text_lower.find(first)
        if start_idx < 0:
            return None, None
        last_idx = text_lower.find(last, start_idx + len(first))
        if last_idx < 0:
            return None, None
        return start_idx, last_idx + len(last)

    def _fuzzy(self, quote: str, doc: Document) -> ValidationResult | None:
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return None
        score = fuzz.partial_ratio(normalize_for_match(quote), normalize_for_match(doc.full_text))
        if score < self.config.fuzzy_threshold:
            return None
        cs, ce = self._fuzzy_locate(quote, doc.full_text)
        ps, pe, section = (None, None, None)
        if cs is not None and ce is not None:
            ps, pe, section = locate(doc, cs, ce)
        return ValidationResult(
            status="validated",
            match_method="fuzzy",
            similarity_score=float(score) / 100.0,
            located_char_start=cs,
            located_char_end=ce,
            located_page_start=ps,
            located_page_end=pe,
            located_section=section,
            located_doc_id=doc.doc_id,
            notes=f"rapidfuzz partial_ratio={score:.1f}",
        )

    @staticmethod
    def _fuzzy_locate(quote: str, full_text: str) -> tuple[int | None, int | None]:
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return None, None
        n_quote = normalize_for_match(quote)
        if not n_quote:
            return None, None
        window = max(len(quote), 40)
        best_score = -1.0
        best_start = -1
        step = max(1, window // 4)
        for i in range(0, max(1, len(full_text) - window + 1), step):
            cand = full_text[i : i + window]
            score = fuzz.partial_ratio(n_quote, normalize_for_match(cand))
            if score > best_score:
                best_score = score
                best_start = i
        if best_start < 0:
            return None, None
        return best_start, min(best_start + window, len(full_text))

    def _semantic(self, quote: str, doc: Document) -> ValidationResult | None:
        score_fn = self.config.semantic_score_fn or self._default_semantic_score
        try:
            score = score_fn(quote, doc.full_text)
        except Exception:
            return None
        if score < self.config.semantic_threshold:
            return None
        return ValidationResult(
            status="semantic",
            match_method="semantic",
            similarity_score=float(score),
            located_doc_id=doc.doc_id,
            notes="Semantic match only — quote was paraphrased, not copied",
        )

    @staticmethod
    def _default_semantic_score(quote: str, text: str) -> float:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        if not hasattr(Validator._default_semantic_score, "_model"):
            Validator._default_semantic_score._model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore[attr-defined]
        model = Validator._default_semantic_score._model  # type: ignore[attr-defined]
        emb = model.encode([quote, text], normalize_embeddings=True, show_progress_bar=False)
        return float((emb[0] * emb[1]).sum())
