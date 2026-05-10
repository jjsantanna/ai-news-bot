"""Text normalization for matching: case, whitespace, smart punctuation.

We deliberately do NOT collapse to the same level as the chunker's
normalization — the validator needs to map normalized matches back to
char offsets in the original text, so the normalization is a 1:1
character mapping (smart-quotes → straight, NBSP → space, etc.) rather
than a destructive transform.
"""
from __future__ import annotations

import re
import unicodedata

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
    "…": "...",
}

_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(_PUNCT_MAP.get(ch, ch) for ch in text)
    text = text.lower()
    text = _WS_RE.sub(" ", text)
    return text.strip()
