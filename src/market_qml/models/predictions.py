"""Standard prediction table formatting for model outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "symbol",
    "date",
    "y_true",
    "y_score",
    "forward_return",
    "forward_excess_return",
    "model_name",
    "split_id",
]


def build_prediction_table(
    *,
    metadata: pd.DataFrame,
    y_true: pd.Series,
    y_score,
    model_name: str,
    split_id: int,
) -> pd.DataFrame:
    """Build the shared prediction table consumed by metrics and backtests."""
    required_metadata = {"symbol", "date"}
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(
            "Validation metadata is missing required columns: "
            + ", ".join(sorted(missing_metadata))
        )

    forward_return_column = _find_forward_return_column(metadata)
    forward_excess_return_column = _find_forward_excess_return_column(metadata)

    result = metadata[["symbol", "date"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if result["date"].isna().any():
        raise ValueError("Validation metadata contains invalid dates.")

    result["y_true"] = pd.to_numeric(y_true, errors="coerce").to_numpy()
    result["y_score"] = pd.to_numeric(pd.Series(y_score), errors="coerce").to_numpy()
    result["forward_return"] = pd.to_numeric(
        metadata[forward_return_column],
        errors="coerce",
    ).to_numpy()
    result["forward_excess_return"] = pd.to_numeric(
        metadata[forward_excess_return_column],
        errors="coerce",
    ).to_numpy()
    result["model_name"] = model_name
    result["split_id"] = split_id

    numeric_columns = [
        "y_true",
        "y_score",
        "forward_return",
        "forward_excess_return",
    ]
    if result[numeric_columns].isna().any().any():
        raise ValueError("Prediction table contains missing or non-numeric values.")

    return (
        result[REQUIRED_PREDICTION_COLUMNS]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a standard prediction table to parquet."""
    _validate_prediction_columns(predictions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _find_forward_return_column(metadata: pd.DataFrame) -> str:
    if "forward_return" in metadata.columns:
        return "forward_return"

    candidates = [
        column
        for column in metadata.columns
        if column.startswith("forward_return_")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Validation metadata must contain exactly one forward return column."
        )
    return candidates[0]


def _find_forward_excess_return_column(metadata: pd.DataFrame) -> str:
    if "forward_excess_return" in metadata.columns:
        return "forward_excess_return"

    candidates = [
        column
        for column in metadata.columns
        if column.startswith("forward_excess_return_")
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Validation metadata must contain exactly one forward excess return column."
        )
    return candidates[0]
