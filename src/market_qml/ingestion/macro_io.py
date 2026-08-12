"""Persistence boundary for macroeconomic ingestion outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_macro_outputs(
    *,
    bls_raw: pd.DataFrame,
    fed_raw: pd.DataFrame,
    combined_raw: pd.DataFrame,
    clean: pd.DataFrame,
    bls_path: Path,
    fed_path: Path,
    combined_path: Path,
    processed_path: Path,
) -> None:
    for path in (bls_path, fed_path, combined_path, processed_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    bls_raw.to_parquet(bls_path, index=False)
    fed_raw.to_parquet(fed_path, index=False)
    combined_raw.to_parquet(combined_path, index=False)
    clean.to_parquet(processed_path)
