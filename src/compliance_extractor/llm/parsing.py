"""Parse JSON-shaped LLM output into an ExtractionResponse.

Models occasionally wrap JSON in markdown fences or prepend text; this
extracts the first JSON object and validates the four required fields.
"""
from __future__ import annotations

import json
import re
from typing import Any

from compliance_extractor.llm.base import Decision

_VALID_DECISIONS = {"complies", "not-complies", "maybe-complies", "no-evidence"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unterminated JSON object in model output: {text[:200]!r}")


def parse_extraction_payload(text: str) -> tuple[Decision, str, str, str]:
    data = extract_json_object(text)
    decision = data.get("decision", "")
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"Invalid decision {decision!r}; expected one of {_VALID_DECISIONS}")
    quote = str(data.get("evidence_quote", ""))
    source = str(data.get("evidence_quote_source", ""))
    rationale = str(data.get("rationale", ""))
    return decision, quote, source, rationale  # type: ignore[return-value]
