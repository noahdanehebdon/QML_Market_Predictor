import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.models.ridge_regression import (
    MODEL_NAME,
    save_predictions,
    save_ridge_regression_model,
    train_ridge_regression,
)


def _dataset(
    X: pd.DataFrame,
    y: list[float],
    *,
    start: str,
    symbols: list[str],
) -> ModelingDataset:
    return ModelingDataset(
        X=X,
        y=pd.Series(y, name="forward_excess_return_5d"),
        metadata=pd.DataFrame(
            {
                "symbol": symbols,
                "date": pd.to_datetime([start] * len(X)),
                "forward_return_5d": [0.02, -0.01, 0.03, 0.01][: len(X)],
                "forward_excess_return_5d": y,
            }
        ),
    )


def _preprocessed() -> TrainValidationDatasets:
    train = _dataset(
        pd.DataFrame(
            {
                "momentum": [-2.0, -1.0, 1.0, 2.0],
                "volatility": [0.5, 0.4, 0.3, 0.2],
            }
        ),
        [-0.04, -0.02, 0.02, 0.04],
        start="2024-01-01",
        symbols=["AAPL", "MSFT", "NVDA", "AMZN"],
    )
    validation = _dataset(
        pd.DataFrame(
            {
                "momentum": [-1.5, 1.5, 0.0],
                "volatility": [0.45, 0.25, 0.35],
            }
        ),
        [-0.03, 0.03, 0.0],
        start="2024-02-01",
        symbols=["AAPL", "MSFT", "NVDA"],
    )
    return fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )


def test_train_ridge_regression_outputs_standard_prediction_table():
    result = train_ridge_regression(_preprocessed())

    assert isinstance(result.model, Ridge)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["split_id"].unique().tolist() == [0]
    assert result.predictions["forward_excess_return"].tolist() == [-0.03, 0.03, 0.0]
    assert result.predictions["y_score"].dtype.kind == "f"


def test_train_ridge_regression_rejects_missing_training_targets():
    data = _preprocessed()
    bad_train = ModelingDataset(
        X=data.train.X,
        y=pd.Series([0.1, None, 0.2, 0.3]),
        metadata=data.train.metadata,
    )

    with pytest.raises(ValueError, match="missing or non-numeric"):
        train_ridge_regression(
            TrainValidationDatasets(train=bad_train, validation=data.validation)
        )


def test_ridge_regression_outputs_can_be_saved(tmp_path):
    result = train_ridge_regression(_preprocessed())
    model_path = tmp_path / "model.pkl"
    prediction_path = tmp_path / "predictions.parquet"

    save_ridge_regression_model(result.model, model_path)
    save_predictions(result.predictions, prediction_path)
    saved_predictions = pd.read_parquet(prediction_path)

    assert model_path.exists()
    assert prediction_path.exists()
    assert list(saved_predictions.columns) == list(result.predictions.columns)
