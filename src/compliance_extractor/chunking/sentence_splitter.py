"""Sentence splitter with char-offset preservation.

Default backend is blingfire (fast, statistical). Falls back to a regex
splitter when blingfire isn't installed so unit tests stay portable.
"""
from __future__ import annotations

import re

_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str, backend: str = "blingfire") -> list[tuple[str, int, int]]:
    """Return [(sentence_text, char_start, char_end)]; offsets index into `text`."""
    if not text:
        return []

    if backend == "blingfire":
        try:
            return _blingfire_split(text)
        except ImportError:
            pass  # fall through to regex
    return _regex_split(text)


def _blingfire_split(text: str) -> list[tuple[str, int, int]]:
    import blingfire  # type: ignore[import-not-found]

    raw = blingfire.text_to_sentences(text)
    sentences = [s for s in raw.split("\n") if s.strip()]
    return _align_to_text(sentences, text)


def _regex_split(text: str) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    last = 0
    for m in _FALLBACK_RE.finditer(text):
        end = m.start()
        seg = text[last:end].strip()
        if seg:
            char_start = text.find(seg, last)
            parts.append((seg, char_start, char_start + len(seg)))
        last = m.end()
    tail = text[last:].strip()
    if tail:
        char_start = text.find(tail, last)
        parts.append((tail, char_start, char_start + len(tail)))
    return parts


def _align_to_text(sentences: list[str], text: str) -> list[tuple[str, int, int]]:
    """Walk through `text` finding each sentence in order to recover offsets."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        idx = text.find(s, cursor)
        if idx < 0:
            idx = text.find(s)
        if idx < 0:
            continue
        out.append((s, idx, idx + len(s)))
        cursor = idx + len(s)
    return out
