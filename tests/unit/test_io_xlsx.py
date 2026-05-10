"""Roundtrip tests for the xlsx I/O helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from compliance_extractor import schemas
from compliance_extractor.config import RunConfig
from compliance_extractor.io.controls_xlsx import read_controls, write_controls
from compliance_extractor.io.results_xlsx import read_evidence, write_evidence


@pytest.mark.fast
def test_controls_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "control_id": "A.5.1",
                "title": "Policies",
                "description": "",
                "theme": "Organizational",
                "standard": "ISO 27001:2022",
            }
        ]
    )
    out = tmp_path / "ctrl.xlsx"
    write_controls(df, out)
    back = read_controls(out)
    assert list(back.columns) == schemas.CONTROLS.columns
    assert back.iloc[0]["control_id"] == "A.5.1"


@pytest.mark.fast
def test_evidence_roundtrip_writes_run_config_sheet(tmp_path: Path) -> None:
    cfg = RunConfig(variant_id="v_test", description="unit test variant")
    ev = schemas.empty(schemas.EVIDENCE)
    out = tmp_path / "ev.xlsx"
    write_evidence(ev, cfg, out, extra_meta={"run_id": "abc", "cost_usd": 0.0})
    back = read_evidence(out)
    assert list(back.columns) == schemas.EVIDENCE.columns
    meta = pd.read_excel(out, sheet_name="run_config")
    assert meta.iloc[0]["variant_id"] == "v_test"
    assert "config_json" in meta.columns
