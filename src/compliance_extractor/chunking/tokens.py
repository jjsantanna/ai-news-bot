"""Token-count estimator for chunk budgeting.

Default backend is `char_estimate` (chars / 4) which OpenAI documents as a good
rule of thumb for English and is fully offline. tiktoken can be used when its
encoding cache is available locally; if the encoding file is missing and the
sandbox blocks the download, we fall back to char_estimate so the pipeline
stays usable.

Chunk budgets are about consistency, not exactness — variants in the eval
grid are compared against each other under the same backend, so a heuristic
count is sufficient for chunk-size decisions.
"""
from __future__ import annotations

from functools import lru_cache

DEFAULT_BACKEND = "char_estimate"
_CHARS_PER_TOKEN = 4.0


def count_tokens(text: str, backend: str = DEFAULT_BACKEND) -> int:
    if not text:
        return 0
    if backend == "char_estimate":
        return max(1, int(round(len(text) / _CHARS_PER_TOKEN)))
    if backend == "word_estimate":
        return max(1, int(round(len(text.split()) * 1.3)))
    if backend.startswith("tiktoken:"):
        encoding = backend.split(":", 1)[1] or "cl100k_base"
        enc = _get_tiktoken(encoding)
        if enc is None:
            return count_tokens(text, "char_estimate")
        return len(enc.encode(text, disallowed_special=()))
    raise ValueError(f"Unknown token-count backend: {backend!r}")


@lru_cache(maxsize=4)
def _get_tiktoken(name: str):
    """Return a cached tiktoken encoder, or None if unavailable offline."""
    try:
        import tiktoken  # type: ignore[import-not-found]

        return tiktoken.get_encoding(name)
    except Exception:
        return None
