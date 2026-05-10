"""Match a predicted evidence row to gold rows.

Two evidences match if they share `control_id` AND their quotes are
similar enough by the rapidfuzz partial_ratio (with a fallback to a
character-overlap heuristic when rapidfuzz isn't installed). The
matcher is deliberately simple — annotators rarely write the *exact*
same quote, so we accept any high-overlap quote within the same
control as a true positive.
"""
from __future__ import annotations

from dataclasses import dataclass

from compliance_extractor.validation.normalize import normalize_for_match


@dataclass
class MatchResult:
    is_match: bool
    similarity: float


def quote_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n_a = normalize_for_match(a)
    n_b = normalize_for_match(b)
    try:
        from rapidfuzz import fuzz

        return float(fuzz.partial_ratio(n_a, n_b)) / 100.0
    except ImportError:
        if n_a in n_b or n_b in n_a:
            return 1.0
        return _char_overlap(n_a, n_b)


def match(predicted_quote: str, gold_quote: str, threshold: float = 0.8) -> MatchResult:
    sim = quote_similarity(predicted_quote, gold_quote)
    return MatchResult(is_match=sim >= threshold, similarity=sim)


def _char_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
