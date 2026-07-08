"""Ridge regression baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import Ridge

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
    alpha: float = 1.0,
) -> RidgeRegressionResult:
    """Train ridge regression on one preprocessed train/validation split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training targets contain missing or non-numeric values.")

    model = Ridge(alpha=alpha)
    model.fit(data.train.X, y_train.astype("float64"))

    y_score = model.predict(data.validation.X)
    predictions = _prediction_frame(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
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
    result["y_true"] = pd.to_numeric(y_true, errors="coerce").astype("float64").to_numpy()
    result["y_score"] = y_score
    result["rank"] = result.groupby("date")["y_score"].rank(
        method="first",
        ascending=False,
    )
    result["model"] = model_name

    return result.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)
