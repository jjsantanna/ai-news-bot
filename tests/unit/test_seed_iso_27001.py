"""Sanity check the ISO 27001 control seed."""
from __future__ import annotations

import pytest

from scripts.seed_iso_27001 import ANNEX_A_CONTROLS, build_dataframe


@pytest.mark.fast
def test_seed_has_93_controls() -> None:
    assert len(ANNEX_A_CONTROLS) == 93


@pytest.mark.fast
def test_seed_control_ids_unique() -> None:
    ids = [c[0] for c in ANNEX_A_CONTROLS]
    assert len(ids) == len(set(ids))


@pytest.mark.fast
def test_seed_themes_are_canonical() -> None:
    themes = {c[1] for c in ANNEX_A_CONTROLS}
    assert themes == {"Organizational", "People", "Physical", "Technological"}


@pytest.mark.fast
def test_build_dataframe_dtypes() -> None:
    df = build_dataframe()
    assert len(df) == 93
    assert str(df["theme"].dtype) == "category"
    assert (df["standard"] == "ISO 27001:2022").all()
