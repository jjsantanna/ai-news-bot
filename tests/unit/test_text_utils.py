"""Text-normalization helpers."""
from __future__ import annotations

import pytest

from compliance_extractor.utils.text import (
    normalize_text,
    normalize_whitespace,
    sliding_windows,
)


@pytest.mark.fast
def test_normalize_whitespace_collapses_runs() -> None:
    assert normalize_whitespace("  a\t\nb   c  ") == "a b c"


@pytest.mark.fast
def test_normalize_text_handles_smart_quotes_and_nfkc() -> None:
    raw = "ﬁle “quote”  here"  # NFKC will split the ligature, NBSP -> space
    assert normalize_text(raw) == 'file "quote" here'


@pytest.mark.fast
def test_sliding_windows_covers_full_text() -> None:
    text = "abcdefghij"
    windows = list(sliding_windows(text, window=4, stride=3))
    assert windows[0] == (0, 4, "abcd")
    assert windows[-1][1] == len(text)
    joined = "".join(w[2] for w in windows)
    assert "j" in joined  # last char included


@pytest.mark.fast
def test_sliding_windows_validates_args() -> None:
    with pytest.raises(ValueError):
        list(sliding_windows("abc", window=0, stride=1))
