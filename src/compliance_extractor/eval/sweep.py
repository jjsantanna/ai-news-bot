"""Sweep runner: run every variant against a fixed corpus + gold, aggregate.

Designed so the same code path works in tests (with a mock LLM factory)
and in production (with real LLM clients). The `llm_factory` callable
receives a RunConfig and returns an LLMClient — variant-specific
construction (Anthropic vs OpenAI vs Ollama, model id, prompt cache)
lives there.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from compliance_extractor.config import RunConfig
from compliance_extractor.eval.metrics import evaluate, metrics_to_row
from compliance_extractor.llm.base import LLMClient
from compliance_extractor.pipeline import run_pipeline


def run_sweep(
    variants: list[RunConfig],
    corpus_paths: list[str | Path],
    controls_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    llm_factory: Callable[[RunConfig], LLMClient],
    *,
    output_dir: str | Path | None = None,
    quote_match_threshold: float = 0.8,
) -> pd.DataFrame:
    """Run every variant; return a DataFrame indexed by variant_id with metrics + per-variant evidence path."""
    rows: list[dict[str, object]] = []
    for variant in variants:
        llm = llm_factory(variant)
        evidence_df = run_pipeline(variant, corpus_paths, controls_df, llm=llm)
        if output_dir:
            out_path = Path(output_dir) / f"evidence_{variant.variant_id}.xlsx"
            from compliance_extractor.io.results_xlsx import write_evidence

            write_evidence(evidence_df, variant, out_path)

        metrics = evaluate(evidence_df, gold_df, quote_match_threshold=quote_match_threshold)
        rows.append(metrics_to_row(variant.variant_id, metrics))

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
