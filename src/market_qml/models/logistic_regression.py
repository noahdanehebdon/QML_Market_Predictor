"""Logistic regression baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

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

    predictions = _prediction_frame(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)


def _prediction_frame(
    *,
    metadata: pd.DataFrame,
    y_true: pd.Series,
    y_score,
    model_name: str,
) -> pd.DataFrame:
    required_metadata = {"symbol", "date"}
    missing_metadata = required_metadata - set(metadata.columns)
    if missing_metadata:
        raise ValueError(
            "Validation metadata is missing required columns: "
            + ", ".join(sorted(missing_metadata))
        )

    result = metadata[["symbol", "date"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["y_true"] = pd.to_numeric(y_true, errors="coerce").astype("Int64").to_numpy()
    result["y_score"] = y_score
    result["model"] = model_name

    return result.sort_values(["symbol", "date"]).reset_index(drop=True)
