"""Read and write the standards controls workbook."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from compliance_extractor import schemas


def read_controls(path: str | Path) -> pd.DataFrame:
    """Read a standards xlsx into a controls DataFrame.

    Expected sheet name: `controls`. The first row is treated as a header.
    """
    df = pd.read_excel(path, sheet_name="controls")
    return schemas.cast(df, schemas.CONTROLS)


def write_controls(df: pd.DataFrame, path: str | Path) -> None:
    """Write a controls DataFrame to xlsx (sheet name `controls`)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = schemas.cast(df, schemas.CONTROLS)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="controls", index=False)
