"""Variant config loading.

A `RunConfig` describes one cell of the experiment grid: chunker + retriever +
extractor + LLM + validator ladder. YAML files in `configs/variants/*.yaml`
deserialize into this model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """One variant of the extraction pipeline.

    Loaded from `configs/variants/*.yaml`. All fields are required at run time;
    defaults reflect the recommended baseline (hybrid + rerank + Claude Opus).
    """

    variant_id: str
    description: str = ""

    chunker: Literal["sentence", "paragraph", "semantic", "section"] = "paragraph"
    chunker_params: dict[str, Any] = Field(default_factory=dict)

    retriever: Literal["bm25", "dense", "hybrid", "none"] = "hybrid"
    embedder: str | None = "bge-m3"
    vector_db: Literal["chromadb"] = "chromadb"
    distance: Literal["cosine"] = "cosine"
    k: int = 5
    rerank: str | None = None

    extractor: Literal["rag", "long_context", "map_reduce", "citations_api"] = "rag"
    llm: str = "claude-opus-4-7"

    validator_ladder: list[Literal["exact", "normalized", "fuzzy", "semantic"]] = Field(
        default_factory=lambda: ["exact", "normalized", "fuzzy"]
    )

    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


def load_variants(directory: str | Path) -> list[RunConfig]:
    """Load every `*.yaml` file in `directory` as a RunConfig."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a variants directory: {directory}")
    return [RunConfig.from_yaml(p) for p in sorted(directory.glob("*.yaml"))]
