"""Typer CLI entry point: `ce <subcommand>`.

Subcommands implemented in v1 bootstrap:
    init-standard   scaffold an empty controls xlsx for a standard
    list-variants   show every YAML found in a variants directory

The remaining subcommands (extract, validate, evaluate, run-sweep) are wired
later as the corresponding modules land.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from compliance_extractor import schemas
from compliance_extractor.config import load_variants
from compliance_extractor.io.controls_xlsx import write_controls

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
    """Run the extraction pipeline (wired once `pipeline.py` lands)."""
    raise typer.Exit(
        code=2,
    )  # pragma: no cover  # placeholder until pipeline is wired


@app.command("validate")
def validate_cmd(
    evidence: Path = typer.Option(..., help="Evidence xlsx to validate."),
    documents: Path = typer.Option(..., help="Directory of source documents."),
    strict: bool = typer.Option(False, help="Strict mode: only exact + normalized rungs."),
    out: Path = typer.Option(..., help="Output evidence xlsx with validation columns filled."),
) -> None:
    """Re-run the validator ladder against an existing evidence xlsx."""
    raise typer.Exit(code=2)  # pragma: no cover


@app.command("evaluate")
def evaluate(
    predictions: Path = typer.Option(..., help="Evidence xlsx (predictions)."),
    groundtruth: Path = typer.Option(..., help="Gold evidence xlsx."),
    out: Path | None = typer.Option(None, help="Optional eval xlsx."),
) -> None:
    """Compute precision/recall/F1/hallucination_rate against the gold set."""
    raise typer.Exit(code=2)  # pragma: no cover


@app.command("run-sweep")
def run_sweep(
    variants: Path = typer.Option(Path("configs/variants"), help="Variants directory."),
    standard: Path = typer.Option(..., help="Controls xlsx."),
    documents: Path = typer.Option(..., help="Directory of source documents."),
    groundtruth: Path = typer.Option(..., help="Gold evidence xlsx."),
    out: Path = typer.Option(..., help="Output sweep directory."),
    max_workers: int = typer.Option(1, help="Parallel variant workers (default 1)."),
) -> None:
    """Run every variant against the gold corpus and emit a comparison xlsx."""
    raise typer.Exit(code=2)  # pragma: no cover


if __name__ == "__main__":
    app()
