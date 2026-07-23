"""Ridge regression baseline model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation

MODEL_NAME = "ridge_regression"
DEFAULT_MODEL_PATH = Path("artifacts/models/ridge_regression.pkl")
DEFAULT_PREDICTION_PATH = Path("data/processed/predictions_ridge_regression.parquet")
DEFAULT_TARGET_COLUMN = "forward_excess_return_5d"


@dataclass(frozen=True)
class RidgeRegressionResult:
    """Fitted model and validation predictions."""

    model: Ridge
    predictions: pd.DataFrame


def train_ridge_regression(
    data: PreprocessedTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int = 0,
    alpha: float = 1.0,
) -> RidgeRegressionResult:
    """Train ridge regression on one preprocessed train/validation split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training targets contain missing or non-numeric values.")

    model = Ridge(alpha=alpha)
    model.fit(data.train.X, y_train.astype("float64"))

    y_score = model.predict(data.validation.X)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
        split_id=split_id,
    )
    return RidgeRegressionResult(model=model, predictions=predictions)


def save_ridge_regression_model(
    model: Ridge,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted ridge regression model."""
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
