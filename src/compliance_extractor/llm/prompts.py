"""System and user prompts shared across providers.

The same prompt text is used across providers so eval-grid comparisons
isolate the provider/model variable. The prompt instructs the model to
return strict JSON with an `evidence_quote` that the validator will
match against the source text.
"""
from __future__ import annotations

from compliance_extractor.llm.base import ExtractionRequest

SYSTEM_PROMPT = """You are an evidence extractor for compliance audits.

Given a control requirement and one or more source documents, decide whether
the documents contain evidence that demonstrates the control is implemented.

Reply with strict JSON matching this schema:

{
  "decision": "complies" | "not-complies" | "maybe-complies" | "no-evidence",
  "evidence_quote": "<a verbatim, contiguous span copied from one source document, or empty string if no-evidence>",
  "evidence_quote_source": "<doc_id of the document the quote was copied from, or empty string>",
  "rationale": "<one or two sentences justifying the decision>"
}

Rules:
1. The `evidence_quote` MUST be a verbatim, contiguous span copied character-for-character from one of the provided documents. Do not paraphrase, summarize, or stitch together discontinuous fragments.
2. The `evidence_quote_source` MUST be the doc_id of the document the quote was taken from.
3. Use "complies" only when the quote directly demonstrates the control is met.
4. Use "not-complies" when the documents explicitly contradict the control.
5. Use "maybe-complies" when the documents partially address the control or are ambiguous.
6. Use "no-evidence" only when the documents contain nothing relevant. In that case, set evidence_quote and evidence_quote_source to "".
7. Output JSON only. No markdown fences, no preamble, no trailing commentary."""


def render_user_prompt(req: ExtractionRequest) -> str:
    parts = [
        f"Control ID: {req.control_id}",
        f"Control title: {req.control_title}",
        f"Control description: {req.control_description}",
    ]
    if req.extra_user_instructions:
        parts.append(f"Extra instructions: {req.extra_user_instructions}")
    parts.append(
        "Decide whether the attached documents contain evidence for this control "
        "and reply with the JSON object specified by the system prompt."
    )
    return "\n".join(parts)
