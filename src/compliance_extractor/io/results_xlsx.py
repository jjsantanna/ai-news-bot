"""Write extraction results (evidence + run-config) to xlsx."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from compliance_extractor import schemas
from compliance_extractor.config import RunConfig


def write_evidence(
    evidence: pd.DataFrame,
    run_cfg: RunConfig,
    path: str | Path,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Write `evidence` + a `run_config` sheet to `path`.

    The run_config sheet is a single-row DataFrame containing the variant id,
    description, full config JSON, and any extra metadata (run_id, cost, etc).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    evidence = schemas.cast(evidence, schemas.EVIDENCE)
    meta_row = {
        "variant_id": run_cfg.variant_id,
        "description": run_cfg.description,
        "config_json": json.dumps(run_cfg.model_dump(), sort_keys=True),
    }
    if extra_meta:
        for k, v in extra_meta.items():
            meta_row[k] = json.dumps(v) if not isinstance(v, (str, int, float, bool)) else v
    meta_df = pd.DataFrame([meta_row])
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        evidence.to_excel(w, sheet_name="evidence", index=False)
        meta_df.to_excel(w, sheet_name="run_config", index=False)


def read_evidence(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="evidence")
    return schemas.cast(df, schemas.EVIDENCE)
