# compliance-evidence-extractor

Extract verbatim compliance evidence from heterogeneous documents (PDF, DOCX, URL, plaintext) against the controls defined in an ISO standard (v1: ISO 27001:2022 Annex A). Outputs an Excel workbook of `(control_id, evidence_text, source, location, validated)` rows.

The system is built around swappable strategies (chunking, retrieval, extraction, validation) so research questions can be answered empirically by running the same input through many variants and comparing metrics on a ground-truth dataset.

## Quick start

```bash
uv sync
uv run ce init-standard --name "ISO 27001:2022" --out standards/iso_27001_2022_annex_a.xlsx
uv run ce extract --standard standards/iso_27001_2022_annex_a.xlsx \
                  --document data/inputs/policy.pdf \
                  --variant configs/variants/v08_citations_api_claude.yaml \
                  --out outputs/runs/run_abc/evidence.xlsx
uv run ce evaluate --predictions outputs/runs/run_abc/evidence.xlsx \
                   --groundtruth data/groundtruth/iso_27001_2022_groundtruth.xlsx
uv run ce run-sweep --variants configs/variants/ --standard ... --documents ... \
                    --groundtruth ... --out outputs/sweeps/2026-05-09/
```

## Layout

See `src/compliance_extractor/` for the package, `configs/variants/` for the variant grid, `standards/` for the controls workbooks, `data/groundtruth/` for the gold evidence annotations, and `tests/` for the suite.

## Research questions answered by the variant sweep

1. RAG (chunking, embedder, K, retrieval mode, rerank) vs long-context LLM — which wins on policy-text evidence extraction?
2. Faithfulness — Anthropic Citations API vs structured-output-with-offsets — does either eliminate hallucinated quotes?
3. Validation — exact / normalized / fuzzy / semantic ladder — what's the precision of each rung?
