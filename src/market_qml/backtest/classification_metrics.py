"""Classification metrics for standard model prediction tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


CLASSIFICATION_METRIC_COLUMNS = [
    "model_name",
    "scope",
    "split_id",
    "rows",
    "positive_labels",
    "positive_rate",
    "mean_score",
    "calibration_gap",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "average_precision",
    "brier_score",
]


def evaluate_classification_metrics(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute binary classification metrics by split and overall."""
    _validate_prediction_table(predictions)
    _validate_threshold(threshold)

    rows = []
    for (model_name, split_id), split_predictions in predictions.groupby(
        ["model_name", "split_id"],
        sort=True,
    ):
        rows.append(
            _metric_row(
                split_predictions,
                model_name=str(model_name),
                split_id=int(split_id),
                scope="split",
                threshold=threshold,
            )
        )

    for model_name, model_predictions in predictions.groupby("model_name", sort=True):
        rows.append(
            _metric_row(
                model_predictions,
                model_name=str(model_name),
                split_id=pd.NA,
                scope="overall",
                threshold=threshold,
            )
        )

    return pd.DataFrame(rows, columns=CLASSIFICATION_METRIC_COLUMNS)


def load_prediction_tables(
    prediction_paths: list[str | Path],
    *,
    skip_non_binary: bool = True,
) -> pd.DataFrame:
    """Load and combine standard prediction parquet files for classification metrics."""
    if not prediction_paths:
        raise ValueError("At least one prediction path is required.")

    frames = []
    skipped_paths = []
    for prediction_path in prediction_paths:
        prediction_path = Path(prediction_path)
        predictions = pd.read_parquet(prediction_path)
        _validate_prediction_columns(predictions)
        if skip_non_binary and not _has_binary_target(predictions["y_true"]):
            skipped_paths.append(prediction_path)
            continue

        frames.append(predictions)

    if not frames:
        skipped = ", ".join(str(path) for path in skipped_paths)
        raise ValueError(f"No binary prediction tables found. Skipped: {skipped}")

    return pd.concat(frames, ignore_index=True)


def save_classification_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save classification metrics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_path, index=False)


def _metric_row(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split_id,
    scope: str,
    threshold: float,
) -> dict:
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce")
    y_score = pd.to_numeric(predictions["y_score"], errors="coerce")
    if y_true.isna().any() or y_score.isna().any():
        raise ValueError("Prediction table contains missing or non-numeric y values.")

    if not _has_binary_target(y_true):
        raise ValueError("Classification metrics require binary y_true values.")

    y_true = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)
    positive_rate = float(y_true.mean())
    mean_score = float(y_score.mean())

    return {
        "model_name": model_name,
        "scope": scope,
        "split_id": split_id,
        "rows": len(predictions),
        "positive_labels": int(y_true.sum()),
        "positive_rate": positive_rate,
        "mean_score": mean_score,
        "calibration_gap": mean_score - positive_rate,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_true, y_score)
            if y_true.nunique() > 1
            else pd.NA
        ),
        "average_precision": average_precision_score(y_true, y_score),
        "brier_score": brier_score_loss(y_true, y_score),
    }


def _validate_prediction_table(predictions: pd.DataFrame) -> None:
    _validate_prediction_columns(predictions)
    if predictions.empty:
        raise ValueError("Prediction table is empty.")
    if predictions[["model_name", "split_id"]].isna().any().any():
        raise ValueError("Prediction table contains missing model_name or split_id values.")


def _validate_prediction_columns(predictions: pd.DataFrame) -> None:
    missing_columns = set(REQUIRED_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")


def _has_binary_target(y_true: pd.Series) -> bool:
    values = pd.to_numeric(y_true, errors="coerce").dropna().unique()
    return len(values) > 0 and set(values).issubset({0, 1})
