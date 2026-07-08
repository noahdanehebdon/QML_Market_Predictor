"""Ranking metrics for standard model prediction tables."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


RANKING_METRIC_COLUMNS = [
    "model_name",
    "scope",
    "split_id",
    "date",
    "rows",
    "information_coefficient",
    "rank_information_coefficient",
    "top_decile_return",
    "bottom_decile_return",
    "long_short_spread",
]


def evaluate_ranking_metrics(
    predictions: pd.DataFrame,
    *,
    return_column: str = "forward_excess_return",
    top_fraction: float = 0.1,
) -> pd.DataFrame:
    """Compute ranking metrics by date, by split, and overall."""
    _validate_prediction_table(predictions, return_column=return_column)
    _validate_top_fraction(top_fraction)

    date_rows = []
    grouped = predictions.groupby(["model_name", "split_id", "date"], sort=True)
    for (model_name, split_id, date), group in grouped:
        date_rows.append(
            _metric_row(
                group,
                model_name=str(model_name),
                scope="date",
                split_id=int(split_id),
                date=pd.Timestamp(date).normalize(),
                return_column=return_column,
                top_fraction=top_fraction,
            )
        )

    date_metrics = pd.DataFrame(date_rows, columns=RANKING_METRIC_COLUMNS)
    aggregate_rows = []
    for (model_name, split_id), group in date_metrics.groupby(
        ["model_name", "split_id"],
        sort=True,
    ):
        aggregate_rows.append(
            _aggregate_row(
                group,
                model_name=str(model_name),
                scope="split",
                split_id=int(split_id),
            )
        )

    for model_name, group in date_metrics.groupby("model_name", sort=True):
        aggregate_rows.append(
            _aggregate_row(
                group,
                model_name=str(model_name),
                scope="overall",
                split_id=pd.NA,
            )
        )

    return pd.concat(
        [date_metrics, pd.DataFrame(aggregate_rows, columns=RANKING_METRIC_COLUMNS)],
        ignore_index=True,
    )


def load_prediction_tables(prediction_paths: list[str | Path]) -> pd.DataFrame:
    """Load and combine standard prediction parquet files for ranking metrics."""
    if not prediction_paths:
        raise ValueError("At least one prediction path is required.")

    frames = []
    for prediction_path in prediction_paths:
        predictions = pd.read_parquet(Path(prediction_path))
        _validate_prediction_columns(predictions)
        frames.append(predictions)

    return pd.concat(frames, ignore_index=True)


def save_ranking_metrics(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Save ranking metrics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_path, index=False)


def _metric_row(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    scope: str,
    split_id,
    date,
    return_column: str,
    top_fraction: float,
) -> dict:
    y_score = pd.to_numeric(predictions["y_score"], errors="coerce")
    returns = pd.to_numeric(predictions[return_column], errors="coerce")
    if y_score.isna().any() or returns.isna().any():
        raise ValueError("Ranking metrics require numeric, non-missing scores and returns.")

    ordered = predictions.assign(
        _score=y_score.to_numpy(),
        _return=returns.to_numpy(),
    ).sort_values("_score", ascending=False)
    tail_count = max(1, math.ceil(len(ordered) * top_fraction))
    top_return = float(ordered.head(tail_count)["_return"].mean())
    bottom_return = float(ordered.tail(tail_count)["_return"].mean())

    return {
        "model_name": model_name,
        "scope": scope,
        "split_id": split_id,
        "date": date,
        "rows": len(ordered),
        "information_coefficient": _safe_corr(y_score, returns, method="pearson"),
        "rank_information_coefficient": _safe_corr(y_score, returns, method="spearman"),
        "top_decile_return": top_return,
        "bottom_decile_return": bottom_return,
        "long_short_spread": top_return - bottom_return,
    }


def _aggregate_row(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    scope: str,
    split_id,
) -> dict:
    metric_columns = [
        "information_coefficient",
        "rank_information_coefficient",
        "top_decile_return",
        "bottom_decile_return",
        "long_short_spread",
    ]
    means = metrics[metric_columns].mean(numeric_only=True)
    return {
        "model_name": model_name,
        "scope": scope,
        "split_id": split_id,
        "date": pd.NaT,
        "rows": int(metrics["rows"].sum()),
        **{column: means[column] for column in metric_columns},
    }


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str):
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return pd.NA
    return x.corr(y, method=method)


def _validate_prediction_table(
    predictions: pd.DataFrame,
    *,
    return_column: str,
) -> None:
    _validate_prediction_columns(predictions)
    if return_column not in predictions.columns:
        raise ValueError(f"Prediction table is missing return column: {return_column}")
    if predictions.empty:
        raise ValueError("Prediction table is empty.")


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _validate_top_fraction(top_fraction: float) -> None:
    if not 0 < top_fraction <= 0.5:
        raise ValueError("top_fraction must be greater than 0 and at most 0.5.")
