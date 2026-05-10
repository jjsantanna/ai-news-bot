"""Typer CLI entry point: `ce <subcommand>`.

Subcommands:
    init-standard   scaffold an empty controls xlsx for a standard
    list-variants   show every YAML found in a variants directory
    extract         run one variant against a corpus, write evidence xlsx
    validate        re-run the validator ladder on an existing evidence xlsx
    evaluate        compute metrics against a gold evidence xlsx
    run-sweep       run every variant in a directory, write a comparison xlsx
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from compliance_extractor import schemas
from compliance_extractor.config import RunConfig, load_variants
from compliance_extractor.io.controls_xlsx import read_controls, write_controls
from compliance_extractor.io.results_xlsx import read_evidence, write_evidence

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command("init-standard")
def init_standard(
    name: str = typer.Option(..., help="Standard name, e.g. 'ISO 27001:2022'."),
    out: Path = typer.Option(..., help="Output xlsx path."),
    seed_iso27001: bool = typer.Option(
        False,
        "--seed-iso27001",
        help="Pre-populate Annex A control ids and titles (93 rows).",
    ),
) -> None:
    """Scaffold an empty controls workbook with the standard's schema."""
    if seed_iso27001:
        from scripts.seed_iso_27001 import build_dataframe  # type: ignore[import-not-found]

        df = build_dataframe(standard=name)
    else:
        df = schemas.empty(schemas.CONTROLS)
    write_controls(df, out)
    console.print(f"[green]Wrote[/green] {out} ({len(df)} controls, standard='{name}')")


@app.command("list-variants")
def list_variants(
    variants_dir: Path = typer.Option(Path("configs/variants"), help="Directory of variant YAMLs."),
) -> None:
    """List every variant config in `variants_dir`."""
    cfgs = load_variants(variants_dir)
    table = Table(title=f"Variants in {variants_dir}")
    for col in ("variant_id", "extractor", "retriever", "embedder", "k", "llm", "rerank"):
        table.add_column(col)
    for c in cfgs:
        table.add_row(
            c.variant_id,
            c.extractor,
            c.retriever,
            c.embedder or "-",
            str(c.k),
            c.llm,
            c.rerank or "-",
        )
    console.print(table)


@app.command("extract")
def extract(
    standard: Path = typer.Option(..., help="Controls xlsx for the standard."),
    document: list[Path] = typer.Option(..., help="One or more input documents."),
    variant: Path = typer.Option(..., help="Path to a variant YAML."),
    out: Path = typer.Option(..., help="Output evidence xlsx."),
) -> None:
    """Run the extraction pipeline for one variant on a list of documents."""
    from compliance_extractor.llm.factory import build_llm_from_config
    from compliance_extractor.pipeline import run_pipeline

    config = RunConfig.from_yaml(variant)
    controls = read_controls(standard)
    llm = build_llm_from_config(config)
    evidence = run_pipeline(config, [str(d) for d in document], controls, llm=llm)
    write_evidence(evidence, config, out)
    console.print(
        f"[green]Wrote[/green] {out}  "
        f"(variant={config.variant_id}, controls={len(controls)}, validated={int(evidence['validated_exists'].sum())})"
    )


@app.command("validate")
def validate_cmd(
    evidence: Path = typer.Option(..., help="Evidence xlsx to validate."),
    documents: Path = typer.Option(..., help="Directory of source documents."),
    variant: Path = typer.Option(..., help="Variant YAML used to produce the evidence."),
    out: Path = typer.Option(..., help="Output evidence xlsx with validation columns refilled."),
    strict: bool = typer.Option(False, help="Strict mode: only exact + normalized rungs."),
) -> None:
    """Re-run the validator ladder against an existing evidence xlsx."""
    from compliance_extractor.ingest.registry import load as load_doc
    from compliance_extractor.validation.ladder import Validator, ValidatorConfig

    cfg = RunConfig.from_yaml(variant)
    df = read_evidence(evidence)
    docs = [load_doc(p) for p in sorted(documents.glob("**/*")) if p.is_file()]
    if not docs:
        console.print(f"[red]No documents found under {documents}[/red]")
        raise typer.Exit(code=2)

    ladder = ["exact", "normalized"] if strict else cfg.validator_ladder
    validator = Validator(ValidatorConfig(enable_semantic="semantic" in ladder))

    rows = []
    for _, row in df.iterrows():
        result = validator.validate(
            evidence_quote=str(row.get("evidence_text", "") or ""),
            evidence_quote_source=str(row.get("doc_id", "") or ""),
            documents=docs,
        )
        new = row.to_dict()
        new["validated_exists"] = result.status == "validated"
        new["match_method"] = result.match_method
        new["match_score"] = float(result.similarity_score)
        new["char_start"] = result.located_char_start
        new["char_end"] = result.located_char_end
        new["page"] = result.located_page_start
        rows.append(new)

    refilled = pd.DataFrame(rows)
    refilled = schemas.cast(refilled, schemas.EVIDENCE)
    write_evidence(refilled, cfg, out)
    console.print(
        f"[green]Wrote[/green] {out}  "
        f"(re-validated {len(refilled)} rows, validated_exists={int(refilled['validated_exists'].sum())})"
    )


@app.command("evaluate")
def evaluate(
    predictions: Path = typer.Option(..., help="Evidence xlsx (predictions)."),
    groundtruth: Path = typer.Option(..., help="Gold evidence xlsx."),
    out: Path | None = typer.Option(None, help="Optional eval xlsx."),
) -> None:
    """Compute precision/recall/F1/hallucination_rate against the gold set."""
    from compliance_extractor.eval.metrics import evaluate as eval_metrics, metrics_to_row

    pred = read_evidence(predictions)
    gold = pd.read_excel(groundtruth, sheet_name=0)
    metrics = eval_metrics(pred, gold)

    table = Table(title=f"Evaluation: {predictions.name} vs {groundtruth.name}")
    for col in ("tp", "fp", "fn", "precision", "recall", "f1", "hallucination_rate"):
        table.add_column(col)
    table.add_row(
        str(metrics.tp),
        str(metrics.fp),
        str(metrics.fn),
        f"{metrics.precision:.3f}",
        f"{metrics.recall:.3f}",
        f"{metrics.f1:.3f}",
        f"{metrics.hallucination_rate:.3f}",
    )
    console.print(table)

    if out:
        df = pd.DataFrame([metrics_to_row(predictions.stem, metrics)])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out, index=False)
        console.print(f"[green]Wrote[/green] {out}")


@app.command("run-sweep")
def run_sweep_cmd(
    variants: Path = typer.Option(Path("configs/variants"), help="Variants directory."),
    standard: Path = typer.Option(..., help="Controls xlsx."),
    documents: Path = typer.Option(..., help="Directory of source documents."),
    groundtruth: Path = typer.Option(..., help="Gold evidence xlsx."),
    out: Path = typer.Option(..., help="Output sweep directory."),
) -> None:
    """Run every variant against the gold corpus and emit a comparison xlsx."""
    from compliance_extractor.eval.sweep import run_sweep
    from compliance_extractor.llm.factory import build_llm_from_config

    cfgs = load_variants(variants)
    controls = read_controls(standard)
    gold = pd.read_excel(groundtruth, sheet_name=0)
    corpus_paths = [str(p) for p in sorted(documents.glob("**/*")) if p.is_file()]

    out.mkdir(parents=True, exist_ok=True)
    sweep_df = run_sweep(
        variants=cfgs,
        corpus_paths=corpus_paths,
        controls_df=controls,
        gold_df=gold,
        llm_factory=build_llm_from_config,
        output_dir=out,
    )
    summary_path = out / "sweep_summary.xlsx"
    sweep_df.to_excel(summary_path, index=False)
    console.print(f"[green]Wrote[/green] {summary_path}  ({len(sweep_df)} variants)")

    table = Table(title="Sweep summary (sorted by F1)")
    for col in sweep_df.columns:
        table.add_column(str(col))
    for _, row in sweep_df.iterrows():
        table.add_row(*[str(v) if not isinstance(v, float) else f"{v:.3f}" for v in row.values])
    console.print(table)


if __name__ == "__main__":
    app()
