"""Gradient boosting regression baseline model."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation

MODEL_NAME = "gradient_boosting_regressor"
DEFAULT_MODEL_PATH = Path("artifacts/models/gradient_boosting_regressor.pkl")
DEFAULT_PREDICTION_PATH = Path(
    "data/processed/predictions_gradient_boosting_regressor.parquet"
)
DEFAULT_PARAMETERS_PATH = Path(
    "artifacts/models/gradient_boosting_regressor_parameters.json"
)
DEFAULT_TARGET_COLUMN = "forward_excess_return_5d"


@dataclass(frozen=True)
class GradientBoostingRegressorResult:
    """Fitted model, validation predictions, and training parameters."""

    model: HistGradientBoostingRegressor
    predictions: pd.DataFrame
    parameters: dict[str, Any]


def train_gradient_boosting_regressor(
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
) -> GradientBoostingRegressorResult:
    """Train histogram gradient boosting regressor on one preprocessed split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training targets contain missing or non-numeric values.")

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
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        min_samples_leaf=min_samples_leaf,
        max_bins=max_bins,
        random_state=random_state,
    )
    model.fit(data.train.X, y_train.astype("float64"))

    y_score = model.predict(data.validation.X)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
        split_id=split_id,
    )

    return GradientBoostingRegressorResult(
        model=model,
        predictions=predictions,
        parameters=parameters,
    )


def save_gradient_boosting_regressor_model(
    model: HistGradientBoostingRegressor,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted gradient boosting regressor model."""
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
