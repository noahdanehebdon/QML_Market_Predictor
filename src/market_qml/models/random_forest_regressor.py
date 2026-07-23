"""Random forest regression baseline model."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation

MODEL_NAME = "random_forest_regressor"
DEFAULT_MODEL_PATH = Path("artifacts/models/random_forest_regressor.pkl")
DEFAULT_PREDICTION_PATH = Path(
    "data/processed/predictions_random_forest_regressor.parquet"
)
DEFAULT_IMPORTANCE_PATH = Path(
    "data/processed/feature_importance_random_forest_regressor.parquet"
)
DEFAULT_TARGET_COLUMN = "forward_excess_return_5d"


@dataclass(frozen=True)
class RandomForestRegressorResult:
    """Fitted model, validation predictions, and feature importances."""

    model: RandomForestRegressor
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def train_random_forest_regressor(
    data: PreprocessedTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int = 0,
    n_estimators: int = 300,
    max_depth: int | None = 6,
    min_samples_leaf: int = 10,
    max_features: str | int | float | None = "sqrt",
    random_state: int = 42,
) -> RandomForestRegressorResult:
    """Train random forest regressor on one preprocessed split."""
    y_train = pd.to_numeric(data.train.y, errors="coerce")
    if y_train.isna().any():
        raise ValueError("Training targets contain missing or non-numeric values.")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
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
    feature_importance = _feature_importance_frame(
        feature_columns=list(data.train.X.columns),
        importances=model.feature_importances_,
        model_name=model_name,
    )

    return RandomForestRegressorResult(
        model=model,
        predictions=predictions,
        feature_importance=feature_importance,
    )


def save_random_forest_regressor_model(
    model: RandomForestRegressor,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted random forest regressor model."""
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


def save_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: str | Path = DEFAULT_IMPORTANCE_PATH,
) -> None:
    """Save feature importance values to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_parquet(output_path, index=False)


def _feature_importance_frame(
    *,
    feature_columns: list[str],
    importances,
    model_name: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
            "model": model_name,
        }
    )
    result["rank"] = result["importance"].rank(
        method="first",
        ascending=False,
    )
    return result.sort_values(["rank", "feature"]).reset_index(drop=True)
