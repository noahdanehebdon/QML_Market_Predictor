"""Simple portfolio backtests from standard model prediction tables."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


PORTFOLIO_RETURN_COLUMNS = [
    "model_name",
    "split_id",
    "date",
    "selected_count",
    "portfolio_return",
    "benchmark_return",
    "excess_return",
    "cumulative_return",
    "benchmark_cumulative_return",
    "cumulative_excess_return",
]


def run_portfolio_backtest(
    predictions: pd.DataFrame,
    *,
    top_k: int | None = None,
    top_fraction: float = 0.1,
) -> pd.DataFrame:
    """Run an equal-weight long-only backtest from model scores."""
    _validate_prediction_table(predictions)
    _validate_selection(top_k=top_k, top_fraction=top_fraction)

    rows = []
    for (model_name, split_id, date), group in predictions.groupby(
        ["model_name", "split_id", "date"],
        sort=True,
    ):
        selected = _select_names(group, top_k=top_k, top_fraction=top_fraction)
        portfolio_return = float(selected["forward_return"].mean())
        benchmark_return = float(
            (selected["forward_return"] - selected["forward_excess_return"]).mean()
        )
        rows.append(
            {
                "model_name": str(model_name),
                "split_id": int(split_id),
                "date": pd.Timestamp(date).normalize(),
                "selected_count": len(selected),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "excess_return": portfolio_return - benchmark_return,
            }
        )

    result = pd.DataFrame(rows).sort_values(["model_name", "split_id", "date"])
    if result.empty:
        raise ValueError("No portfolio return rows were produced.")

    result["cumulative_return"] = result.groupby(["model_name", "split_id"])[
        "portfolio_return"
    ].transform(lambda returns: (1 + returns).cumprod() - 1)
    result["benchmark_cumulative_return"] = result.groupby(["model_name", "split_id"])[
        "benchmark_return"
    ].transform(lambda returns: (1 + returns).cumprod() - 1)
    result["cumulative_excess_return"] = (
        result["cumulative_return"] - result["benchmark_cumulative_return"]
    )

    return result[PORTFOLIO_RETURN_COLUMNS].reset_index(drop=True)


def load_prediction_tables(prediction_paths: list[str | Path]) -> pd.DataFrame:
    """Load and combine standard prediction parquet files for portfolio backtests."""
    if not prediction_paths:
        raise ValueError("At least one prediction path is required.")

    frames = []
    for prediction_path in prediction_paths:
        predictions = pd.read_parquet(Path(prediction_path))
        _validate_prediction_columns(predictions)
        frames.append(predictions)

    return pd.concat(frames, ignore_index=True)


def save_portfolio_returns(portfolio_returns: pd.DataFrame, output_path: str | Path) -> None:
    """Save portfolio return series to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    portfolio_returns.to_parquet(output_path, index=False)


def _select_names(
    predictions: pd.DataFrame,
    *,
    top_k: int | None,
    top_fraction: float,
) -> pd.DataFrame:
    ordered = predictions.sort_values(["y_score", "symbol"], ascending=[False, True])
    count = top_k if top_k is not None else math.ceil(len(ordered) * top_fraction)
    count = max(1, min(count, len(ordered)))
    return ordered.head(count)


def _validate_prediction_table(predictions: pd.DataFrame) -> None:
    _validate_prediction_columns(predictions)
    if predictions.empty:
        raise ValueError("Prediction table is empty.")

    numeric_columns = ["y_score", "forward_return", "forward_excess_return"]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Portfolio backtest requires numeric, non-missing scores and returns.")


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _validate_selection(*, top_k: int | None, top_fraction: float) -> None:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be greater than 0 and at most 1.")
