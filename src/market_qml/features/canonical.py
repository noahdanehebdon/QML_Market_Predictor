"""Canonical modeling feature table construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_qml.features.cross_sectional import add_cross_sectional_features
from market_qml.features.market_signals import add_market_signal_features

REQUIRED_KEY_COLUMNS = {"symbol", "date"}
LABEL_COLUMN_MARKERS = (
    "forward_return",
    "forward_excess_return",
    "outperform_",
    "label_horizon",
)


def build_canonical_features(
    features: pd.DataFrame,
    *,
    add_cross_sectional: bool = True,
) -> pd.DataFrame:
    """Validate and sort the canonical feature table keyed by symbol/date."""
    missing_columns = REQUIRED_KEY_COLUMNS - set(features.columns)
    if missing_columns:
        raise ValueError(
            "Feature table is missing required key columns: "
            + ", ".join(sorted(missing_columns))
        )

    label_columns = _label_columns(features.columns)
    if label_columns:
        raise ValueError(
            "Feature table contains label columns: " + ", ".join(sorted(label_columns))
        )

    result = features.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    if result["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")

    if {"close", "high", "low", "volume", "return_1d"}.issubset(result):
        result = add_market_signal_features(result)

    if add_cross_sectional:
        result = add_cross_sectional_features(result)

    duplicate_keys = result.duplicated(subset=["symbol", "date"], keep=False)
    if duplicate_keys.any():
        duplicated = (
            result.loc[duplicate_keys, ["symbol", "date"]]
            .drop_duplicates()
            .sort_values(["symbol", "date"])
        )
        examples = [
            f"{row.symbol}/{row.date.date()}"
            for row in duplicated.head(5).itertuples(index=False)
        ]
        raise ValueError(
            "Feature table contains duplicate symbol/date rows: " + ", ".join(examples)
        )

    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_canonical_feature_table(
    feature_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Load the final cumulative feature set, validate it, and save it."""
    feature_path = Path(feature_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}. "
            "Run python -m scripts.build_filing_event_features first."
        )

    features = pd.read_parquet(feature_path)
    result = build_canonical_features(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result


def _label_columns(columns: pd.Index) -> list[str]:
    return [
        column
        for column in columns
        if any(marker in str(column).lower() for marker in LABEL_COLUMN_MARKERS)
    ]
