"""Gradient boosting classifier baseline model."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation

MODEL_NAME = "gradient_boosting"
DEFAULT_MODEL_PATH = Path("artifacts/models/gradient_boosting.pkl")
DEFAULT_PREDICTION_PATH = Path("data/processed/predictions_gradient_boosting.parquet")
DEFAULT_METRICS_PATH = Path("data/processed/metrics_gradient_boosting.parquet")
DEFAULT_PARAMETERS_PATH = Path("artifacts/models/gradient_boosting_parameters.json")


@dataclass(frozen=True)
class GradientBoostingResult:
    """Fitted model, validation predictions, metrics, and training parameters."""

    model: HistGradientBoostingClassifier
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    parameters: dict[str, Any]


def train_gradient_boosting(
    data: PreprocessedTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int = 0,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    max_leaf_nodes: int = 31,
    l2_regularization: float = 0.0,
    min_samples_leaf: int = 20,
    max_bins: int = 255,
    random_state: int = 42,
) -> GradientBoostingResult:
    """Train histogram gradient boosting on one preprocessed split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training labels contain missing or non-numeric values.")

    classes = sorted(y_train.astype(int).unique())
    if len(classes) < 2:
        raise ValueError("Training labels must contain at least two classes.")

    parameters = {
        "model": model_name,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "l2_regularization": l2_regularization,
        "min_samples_leaf": min_samples_leaf,
        "max_bins": max_bins,
        "random_state": random_state,
    }
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        max_bins=max_bins,
        random_state=random_state,
    )
    model.fit(data.train.X, y_train.astype(int))

    positive_class_index = list(model.classes_).index(1)
    y_score = model.predict_proba(data.validation.X)[:, positive_class_index]

    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
        split_id=split_id,
    )
    metrics = _metrics_frame(
        y_true=predictions["y_true"],
        y_score=predictions["y_score"],
        model_name=model_name,
    )

    return GradientBoostingResult(
        model=model,
        predictions=predictions,
        metrics=metrics,
        parameters=parameters,
    )


def save_gradient_boosting_model(
    model: HistGradientBoostingClassifier,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted gradient boosting model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(model, f)


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = DEFAULT_PREDICTION_PATH,
) -> None:
    """Save validation predictions to parquet."""
    save_prediction_table(predictions, output_path)


def save_metrics(
    metrics: pd.DataFrame,
    output_path: str | Path = DEFAULT_METRICS_PATH,
) -> None:
    """Save validation metrics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output_path, index=False)


def save_model_parameters(
    parameters: dict[str, Any],
    output_path: str | Path = DEFAULT_PARAMETERS_PATH,
) -> None:
    """Save model parameters to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(parameters, indent=2, sort_keys=True), encoding="utf-8"
    )


def _metrics_frame(
    *,
    y_true: pd.Series,
    y_score: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    y_true_numeric = pd.to_numeric(y_true, errors="coerce")
    y_score_numeric = pd.to_numeric(y_score, errors="coerce")
    if y_true_numeric.isna().any() or y_score_numeric.isna().any():
        raise ValueError(
            "Validation labels and scores must be numeric and non-missing."
        )

    y_true_int = y_true_numeric.astype(int)
    y_pred = (y_score_numeric >= 0.5).astype(int)
    roc_auc = (
        roc_auc_score(y_true_int, y_score_numeric)
        if y_true_int.nunique() > 1
        else pd.NA
    )

    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "rows": len(y_true_int),
                "positive_labels": int(y_true_int.sum()),
                "positive_rate": float(y_true_int.mean()),
                "roc_auc": roc_auc,
                "average_precision": average_precision_score(
                    y_true_int,
                    y_score_numeric,
                ),
                "accuracy_at_0_5": accuracy_score(y_true_int, y_pred),
                "brier_score": brier_score_loss(y_true_int, y_score_numeric),
            }
        ]
    )
