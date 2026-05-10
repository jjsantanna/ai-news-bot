"""End-to-end orchestration: load -> chunk -> retrieve -> extract -> validate -> emit row.

`run_pipeline(config, corpus_paths, controls_df, llm)` is the single entry
point: it builds the chunker / retriever / extractor / validator from the
RunConfig string fields via the registries, runs the full pipeline once
per control, and returns a DataFrame matching `schemas.EVIDENCE`.

The LLM client is passed in (not built from the config) so callers can
inject mocks for testing and so the same pipeline runs offline against
fakes and online against real APIs without touching pipeline code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from compliance_extractor import schemas
from compliance_extractor.chunking.registry import build_chunker
from compliance_extractor.config import RunConfig
from compliance_extractor.extractors.base import Control, ExtractorContext
from compliance_extractor.extractors.registry import build_extractor
from compliance_extractor.ingest.base import Document
from compliance_extractor.ingest.registry import load as load_document
from compliance_extractor.llm.base import LLMClient
from compliance_extractor.retrieval.registry import build_retriever
from compliance_extractor.schemas import EvidenceCols
from compliance_extractor.validation.ladder import Validator, ValidatorConfig


def run_pipeline(
    config: RunConfig,
    corpus_paths: list[str | Path],
    controls_df: pd.DataFrame,
    llm: LLMClient,
    *,
    documents: list[Document] | None = None,
) -> pd.DataFrame:
    """Run one variant against one corpus across all controls. Returns evidence df."""
    docs = documents if documents is not None else [load_document(p) for p in corpus_paths]
    if not docs:
        raise ValueError("Pipeline requires at least one source document")

    rows: list[dict[str, Any]] = []
    for _, control_row in controls_df.iterrows():
        control = Control(
            control_id=str(control_row["control_id"]),
            title=str(control_row.get("title", "") or ""),
            description=str(control_row.get("description", "") or ""),
        )
        chunker = _build_chunker_or_none(config) if config.extractor != "long_context" else None
        retriever = _build_retriever_or_none(config) if config.extractor in {"rag", "citations_rag"} else None
        extractor = build_extractor(config.extractor)
        ctx = ExtractorContext(
            control=control,
            documents=docs,
            llm=llm,
            chunker=chunker,
            retriever=retriever,
            top_k=config.k,
        )
        response = extractor.extract(ctx)

        validator = _build_validator(config)
        validation = validator.validate(
            evidence_quote=response.evidence_quote,
            evidence_quote_source=response.evidence_quote_source,
            documents=docs,
            citation=response.citation,
        )

        rows.append(_make_row(control, response, validation, config))

    df = pd.DataFrame(rows)
    return schemas.cast(df, schemas.EVIDENCE)


def _build_chunker_or_none(config: RunConfig):
    params = dict(config.chunker_params)
    return build_chunker(config.chunker, **params)


def _build_retriever_or_none(config: RunConfig):
    if config.retriever == "none":
        return None
    if config.retriever == "rerank":
        return build_retriever("rerank", base="hybrid")
    return build_retriever(config.retriever)


def _build_validator(config: RunConfig) -> Validator:
    enable_semantic = "semantic" in config.validator_ladder
    return Validator(ValidatorConfig(enable_semantic=enable_semantic))


def _make_row(
    control: Control,
    response,
    validation,
    config: RunConfig,
) -> dict[str, Any]:
    confidence_tier = _confidence_tier(validation.match_method, validation.similarity_score)
    return {
        EvidenceCols.CONTROL_ID: control.control_id,
        EvidenceCols.DOC_ID: validation.located_doc_id or response.evidence_quote_source,
        EvidenceCols.EVIDENCE_TEXT: response.evidence_quote,
        EvidenceCols.LOCATION: validation.located_section or "",
        EvidenceCols.CHAR_START: validation.located_char_start,
        EvidenceCols.CHAR_END: validation.located_char_end,
        EvidenceCols.CONFIDENCE: float(validation.similarity_score),
        EvidenceCols.EXTRACTOR: config.extractor,
        EvidenceCols.RAW_LLM_OUTPUT: response.rationale,
        EvidenceCols.VALIDATED_EXISTS: validation.status == "validated",
        EvidenceCols.MATCH_METHOD: validation.match_method,
        EvidenceCols.MATCH_SCORE: float(validation.similarity_score),
        EvidenceCols.CONFIDENCE_TIER: confidence_tier,
        EvidenceCols.PAGE: validation.located_page_start,
        EvidenceCols.LINE: None,
    }


def _confidence_tier(method: str, score: float) -> str:
    if method in {"citation", "exact"}:
        return "high"
    if method == "normalized":
        return "high"
    if method == "fuzzy":
        return "medium" if score >= 0.95 else "low"
    if method == "semantic":
        return "low"
    if method == "skipped":
        return "n/a"
    return "none"
