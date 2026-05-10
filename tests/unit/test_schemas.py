"""Schema dtype enforcement and empty-frame helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from compliance_extractor import schemas


@pytest.mark.fast
def test_empty_controls_has_correct_columns_and_dtypes() -> None:
    df = schemas.empty(schemas.CONTROLS)
    assert list(df.columns) == schemas.CONTROLS.columns
    assert len(df) == 0
    assert str(df[schemas.ControlsCols.CONTROL_ID].dtype) == "string"
    assert str(df[schemas.ControlsCols.STANDARD].dtype) == "category"


@pytest.mark.fast
def test_cast_adds_missing_columns_as_nulls() -> None:
    df = pd.DataFrame({schemas.ControlsCols.CONTROL_ID: ["A.5.1"]})
    out = schemas.cast(df, schemas.CONTROLS)
    assert list(out.columns) == schemas.CONTROLS.columns
    assert pd.isna(out[schemas.ControlsCols.TITLE].iloc[0])


@pytest.mark.fast
def test_cast_strict_rejects_extra_columns() -> None:
    df = pd.DataFrame({schemas.ControlsCols.CONTROL_ID: ["A.5.1"], "extra": ["x"]})
    with pytest.raises(ValueError, match="Unexpected columns"):
        schemas.cast(df, schemas.CONTROLS, strict=True)


@pytest.mark.fast
def test_evidence_schema_includes_validation_columns() -> None:
    cols = schemas.EVIDENCE.columns
    for required in (
        "control_id",
        "evidence_text",
        "validated_exists",
        "match_method",
        "confidence_tier",
    ):
        assert required in cols
