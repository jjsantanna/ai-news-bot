"""Text-normalization helpers used by chunkers, validators, and matchers."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_WHITESPACE_RE = re.compile(r"\s+")
_SMART_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_whitespace(s: str) -> str:
    """Collapse all runs of whitespace to a single space; strip ends."""
    return _WHITESPACE_RE.sub(" ", s).strip()


def normalize_text(s: str) -> str:
    """NFKC unicode-normalize, replace smart quotes, collapse whitespace.

    Used by the validator's `normalized` rung and by the metric matcher.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_SMART_QUOTES)
    return normalize_whitespace(s)


def sliding_windows(text: str, *, window: int, stride: int) -> Iterator[tuple[int, int, str]]:
    """Yield (start, end, slice) tuples covering `text` with the given window/stride.

    Window and stride are character counts. The final window is clipped to the
    end of the string.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    n = len(text)
    i = 0
    while i < n:
        j = min(i + window, n)
        yield i, j, text[i:j]
        if j == n:
            return
        i += stride
