import json

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.gradient_boosting_regressor import (
    MODEL_NAME,
    save_gradient_boosting_regressor_model,
    save_model_parameters,
    save_predictions,
    train_gradient_boosting_regressor,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.models.preprocessing import fit_transform_train_validation


def _dataset(
    X: pd.DataFrame,
    y: list[float],
    *,
    start: str,
) -> ModelingDataset:
    return ModelingDataset(
        X=X,
        y=pd.Series(y, name="forward_excess_return_5d"),
        metadata=pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"][: len(X)],
                "date": pd.date_range(start, periods=len(X), freq="D"),
                "forward_return_5d": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04][
                    : len(X)
                ],
                "forward_excess_return_5d": y,
            }
        ),
    )


def _preprocessed() -> TrainValidationDatasets:
    train = _dataset(
        pd.DataFrame(
            {
                "momentum": [-2.0, -1.2, -0.5, 0.6, 1.3, 2.0],
                "volatility": [0.6, 0.5, 0.45, 0.35, 0.25, 0.2],
                "market_beta": [1.2, 1.1, 1.0, 0.95, 0.9, 0.85],
            }
        ),
        [-0.04, -0.03, -0.01, 0.01, 0.02, 0.04],
        start="2024-01-01",
    )
    validation = _dataset(
        pd.DataFrame(
            {
                "momentum": [-1.5, 1.5],
                "volatility": [0.55, 0.22],
                "market_beta": [1.15, 0.88],
            }
        ),
        [-0.03, 0.03],
        start="2024-02-01",
    )
    return fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )


def test_train_gradient_boosting_regressor_outputs_required_prediction_columns():
    result = train_gradient_boosting_regressor(
        _preprocessed(),
        max_iter=20,
        min_samples_leaf=1,
    )

    assert isinstance(result.model, HistGradientBoostingRegressor)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["split_id"].unique().tolist() == [0]
    assert result.predictions["y_true"].tolist() == [-0.03, 0.03]
    assert result.predictions["y_score"].dtype.kind == "f"


def test_train_gradient_boosting_regressor_outputs_parameters():
    result = train_gradient_boosting_regressor(
        _preprocessed(),
        learning_rate=0.1,
        max_iter=20,
        min_samples_leaf=1,
    )

    assert result.parameters["model"] == MODEL_NAME
    assert result.parameters["learning_rate"] == 0.1
    assert result.parameters["max_iter"] == 20
    assert result.parameters["min_samples_leaf"] == 1


def test_train_gradient_boosting_regressor_rejects_missing_training_targets():
    data = _preprocessed()
    bad_train = ModelingDataset(
        X=data.train.X,
        y=pd.Series([0.1, None, 0.2, 0.3, 0.4, 0.5]),
        metadata=data.train.metadata,
    )

    with pytest.raises(ValueError, match="missing or non-numeric"):
        train_gradient_boosting_regressor(
            TrainValidationDatasets(
                train=bad_train,
                validation=data.validation,
            )
        )


def test_gradient_boosting_regressor_outputs_can_be_saved(tmp_path):
    result = train_gradient_boosting_regressor(
        _preprocessed(),
        max_iter=20,
        min_samples_leaf=1,
    )
    model_path = tmp_path / "model.pkl"
    prediction_path = tmp_path / "predictions.parquet"
    parameters_path = tmp_path / "parameters.json"

    save_gradient_boosting_regressor_model(result.model, model_path)
    save_predictions(result.predictions, prediction_path)
    save_model_parameters(result.parameters, parameters_path)
    saved_predictions = pd.read_parquet(prediction_path)
    saved_parameters = json.loads(parameters_path.read_text(encoding="utf-8"))

    assert model_path.exists()
    assert prediction_path.exists()
    assert parameters_path.exists()
    assert list(saved_predictions.columns) == list(result.predictions.columns)
    assert saved_parameters["model"] == MODEL_NAME
