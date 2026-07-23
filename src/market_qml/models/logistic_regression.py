"""Logistic regression baseline model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation

MODEL_NAME = "logistic_regression"
DEFAULT_MODEL_PATH = Path("artifacts/models/logistic_regression.pkl")
DEFAULT_PREDICTION_PATH = Path("data/processed/predictions_logistic_regression.parquet")


@dataclass(frozen=True)
class LogisticRegressionResult:
    """Fitted model and validation predictions."""

    model: LogisticRegression
    predictions: pd.DataFrame


def train_logistic_regression(
    data: PreprocessedTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int = 0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegressionResult:
    """Train logistic regression on one preprocessed train/validation split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training labels contain missing or non-numeric values.")

    classes = sorted(y_train.astype(int).unique())
    if len(classes) < 2:
        raise ValueError("Training labels must contain at least two classes.")

    model = LogisticRegression(
        max_iter=max_iter,
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
    return LogisticRegressionResult(model=model, predictions=predictions)


def save_logistic_regression_model(
    model: LogisticRegression,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted logistic regression model."""
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
