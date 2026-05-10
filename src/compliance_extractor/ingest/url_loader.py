"""URL loader: fetch HTML and extract main text via trafilatura."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

from compliance_extractor.ingest.base import Document, TextBlock


class UrlLoader:
    name = "url"

    def can_handle(self, source: str | Path) -> bool:
        s = str(source)
        return s.startswith("http://") or s.startswith("https://")

    def load(self, source: str | Path) -> Document:
        import httpx
        import trafilatura

        url = str(source)
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url, headers={"User-Agent": "compliance-extractor/0.1"})
            resp.raise_for_status()
            html = resp.text

        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        if not text.endswith("\n"):
            text = text + "\n"

        host = urlparse(url).netloc
        digest = hashlib.sha256(url.encode()).hexdigest()[:8]
        doc_id = f"{host}_{digest}"

        blocks = [TextBlock(text=text, char_start=0, char_end=len(text), page=None, section=None)]
        return Document(
            doc_id=doc_id,
            full_text=text,
            blocks=blocks,
            metadata={"source_url": url, "modality": "url", "http_status": resp.status_code},
        )
