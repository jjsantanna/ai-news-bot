"""DataFrame column-name constants and dtype casters.

Every pandas DataFrame produced by this package conforms to one of the schemas
defined here. Use the helper `cast(df, schema)` to enforce dtypes after a read.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ControlsCols:
    CONTROL_ID = "control_id"
    TITLE = "title"
    DESCRIPTION = "description"
    THEME = "theme"
    STANDARD = "standard"


CONTROLS_DTYPES: dict[str, str] = {
    ControlsCols.CONTROL_ID: "string",
    ControlsCols.TITLE: "string",
    ControlsCols.DESCRIPTION: "string",
    ControlsCols.THEME: "category",
    ControlsCols.STANDARD: "category",
}


class ChunkCols:
    CHUNK_ID = "chunk_id"
    DOC_ID = "doc_id"
    TEXT = "text"
    PAGE = "page"
    SECTION = "section"
    CHAR_START = "char_start"
    CHAR_END = "char_end"
    TOKEN_COUNT = "token_count"


CHUNK_DTYPES: dict[str, str] = {
    ChunkCols.CHUNK_ID: "string",
    ChunkCols.DOC_ID: "string",
    ChunkCols.TEXT: "string",
    ChunkCols.PAGE: "Int64",
    ChunkCols.SECTION: "string",
    ChunkCols.CHAR_START: "Int64",
    ChunkCols.CHAR_END: "Int64",
    ChunkCols.TOKEN_COUNT: "Int32",
}


class CandidateCols:
    CONTROL_ID = "control_id"
    DOC_ID = "doc_id"
    EVIDENCE_TEXT = "evidence_text"
    LOCATION = "location"
    CHAR_START = "char_start"
    CHAR_END = "char_end"
    CONFIDENCE = "confidence"
    EXTRACTOR = "extractor"
    RAW_LLM_OUTPUT = "raw_llm_output"


CANDIDATE_DTYPES: dict[str, str] = {
    CandidateCols.CONTROL_ID: "string",
    CandidateCols.DOC_ID: "string",
    CandidateCols.EVIDENCE_TEXT: "string",
    CandidateCols.LOCATION: "string",
    CandidateCols.CHAR_START: "Int64",
    CandidateCols.CHAR_END: "Int64",
    CandidateCols.CONFIDENCE: "Float32",
    CandidateCols.EXTRACTOR: "category",
    CandidateCols.RAW_LLM_OUTPUT: "string",
}


class EvidenceCols(CandidateCols):
    VALIDATED_EXISTS = "validated_exists"
    MATCH_METHOD = "match_method"
    MATCH_SCORE = "match_score"
    CONFIDENCE_TIER = "confidence_tier"
    PAGE = "page"
    LINE = "line"


EVIDENCE_DTYPES: dict[str, str] = {
    **CANDIDATE_DTYPES,
    EvidenceCols.VALIDATED_EXISTS: "boolean",
    EvidenceCols.MATCH_METHOD: "category",
    EvidenceCols.MATCH_SCORE: "Float32",
    EvidenceCols.CONFIDENCE_TIER: "category",
    EvidenceCols.PAGE: "Int64",
    EvidenceCols.LINE: "Int64",
}


class EvalCols:
    VARIANT_ID = "variant_id"
    CONTROL_ID = "control_id"
    DOC_ID = "doc_id"
    TP = "tp"
    FP = "fp"
    FN = "fn"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    HALLUCINATION_RATE = "hallucination_rate"
    LATENCY_S = "latency_s"
    COST_USD = "cost_usd"


EVAL_DTYPES: dict[str, str] = {
    EvalCols.VARIANT_ID: "category",
    EvalCols.CONTROL_ID: "string",
    EvalCols.DOC_ID: "string",
    EvalCols.TP: "Int32",
    EvalCols.FP: "Int32",
    EvalCols.FN: "Int32",
    EvalCols.PRECISION: "Float32",
    EvalCols.RECALL: "Float32",
    EvalCols.F1: "Float32",
    EvalCols.HALLUCINATION_RATE: "Float32",
    EvalCols.LATENCY_S: "Float32",
    EvalCols.COST_USD: "Float32",
}


class GoldCols:
    GOLD_ID = "gold_id"
    CONTROL_ID = "control_id"
    SOURCE_DOC = "source_doc"
    EXACT_QUOTE = "exact_quote"
    CHAR_START = "char_start"
    CHAR_END = "char_end"
    PAGE = "page"
    SECTION = "section"
    ANNOTATOR = "annotator"
    NOTES = "notes"


GOLD_DTYPES: dict[str, str] = {
    GoldCols.GOLD_ID: "string",
    GoldCols.CONTROL_ID: "string",
    GoldCols.SOURCE_DOC: "string",
    GoldCols.EXACT_QUOTE: "string",
    GoldCols.CHAR_START: "Int64",
    GoldCols.CHAR_END: "Int64",
    GoldCols.PAGE: "Int64",
    GoldCols.SECTION: "string",
    GoldCols.ANNOTATOR: "string",
    GoldCols.NOTES: "string",
}


@dataclass(frozen=True)
class Schema:
    name: str
    dtypes: dict[str, str]

    @property
    def columns(self) -> list[str]:
        return list(self.dtypes.keys())


CONTROLS = Schema("controls", CONTROLS_DTYPES)
CHUNKS = Schema("chunks", CHUNK_DTYPES)
CANDIDATES = Schema("candidates", CANDIDATE_DTYPES)
EVIDENCE = Schema("evidence", EVIDENCE_DTYPES)
EVAL = Schema("eval", EVAL_DTYPES)
GOLD = Schema("gold", GOLD_DTYPES)


def empty(schema: Schema) -> pd.DataFrame:
    """Return an empty DataFrame with the given schema's columns and dtypes."""
    df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in schema.dtypes.items()})
    return df


def cast(df: pd.DataFrame, schema: Schema, *, strict: bool = False) -> pd.DataFrame:
    """Cast `df` to `schema`'s dtypes. Adds missing columns as nulls.

    If `strict`, raises if `df` has columns not in the schema.
    """
    if strict:
        extra = set(df.columns) - set(schema.dtypes)
        if extra:
            raise ValueError(f"Unexpected columns for schema {schema.name}: {sorted(extra)}")
    out = df.copy()
    for col, dt in schema.dtypes.items():
        if col not in out.columns:
            out[col] = pd.Series([pd.NA] * len(out), dtype=dt)
        else:
            out[col] = out[col].astype(dt)
    return out[schema.columns]
