import json

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.gradient_boosting import (
    MODEL_NAME,
    save_gradient_boosting_model,
    save_metrics,
    save_model_parameters,
    save_predictions,
    train_gradient_boosting,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.models.preprocessing import fit_transform_train_validation


def _dataset(
    X: pd.DataFrame,
    y: list[int],
    *,
    start: str,
) -> ModelingDataset:
    return ModelingDataset(
        X=X,
        y=pd.Series(y, name="outperform_spy_5d"),
        metadata=pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"][: len(X)],
                "date": pd.date_range(start, periods=len(X), freq="D"),
                "forward_return_5d": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04][
                    : len(X)
                ],
                "forward_excess_return_5d": [
                    0.01,
                    -0.02,
                    0.02,
                    0.0,
                    -0.03,
                    0.03,
                ][: len(X)],
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
        [0, 0, 0, 1, 1, 1],
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
        [0, 1],
        start="2024-02-01",
    )
    return fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )


def test_train_gradient_boosting_outputs_required_prediction_columns():
    result = train_gradient_boosting(
        _preprocessed(),
        max_iter=20,
        min_samples_leaf=1,
    )

    assert isinstance(result.model, HistGradientBoostingClassifier)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["split_id"].unique().tolist() == [0]
    assert result.predictions["y_true"].tolist() == [0, 1]
    assert result.predictions["y_score"].between(0, 1).all()


def test_train_gradient_boosting_outputs_metrics_and_parameters():
    result = train_gradient_boosting(
        _preprocessed(),
        learning_rate=0.1,
        max_iter=20,
        min_samples_leaf=1,
    )

    assert list(result.metrics.columns) == [
        "model_name",
        "rows",
        "positive_labels",
        "positive_rate",
        "roc_auc",
        "average_precision",
        "accuracy_at_0_5",
        "brier_score",
    ]
    assert result.metrics.loc[0, "model_name"] == MODEL_NAME
    assert result.metrics.loc[0, "rows"] == 2
    assert result.metrics.loc[0, "positive_labels"] == 1
    assert result.metrics.loc[0, "roc_auc"] >= 0
    assert result.parameters["model"] == MODEL_NAME
    assert result.parameters["learning_rate"] == 0.1
    assert result.parameters["max_iter"] == 20


def test_train_gradient_boosting_rejects_single_class_training_labels():
    data = _preprocessed()
    single_class_train = ModelingDataset(
        X=data.train.X,
        y=pd.Series([1] * len(data.train.y)),
        metadata=data.train.metadata,
    )

    with pytest.raises(ValueError, match="at least two classes"):
        train_gradient_boosting(
            TrainValidationDatasets(
                train=single_class_train,
                validation=data.validation,
            )
        )


def test_gradient_boosting_outputs_can_be_saved(tmp_path):
    result = train_gradient_boosting(
        _preprocessed(),
        max_iter=20,
        min_samples_leaf=1,
    )
    model_path = tmp_path / "model.pkl"
    prediction_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "metrics.parquet"
    parameters_path = tmp_path / "parameters.json"

    save_gradient_boosting_model(result.model, model_path)
    save_predictions(result.predictions, prediction_path)
    save_metrics(result.metrics, metrics_path)
    save_model_parameters(result.parameters, parameters_path)
    saved_predictions = pd.read_parquet(prediction_path)
    saved_metrics = pd.read_parquet(metrics_path)
    saved_parameters = json.loads(parameters_path.read_text(encoding="utf-8"))

    assert model_path.exists()
    assert prediction_path.exists()
    assert metrics_path.exists()
    assert parameters_path.exists()
    assert list(saved_predictions.columns) == list(result.predictions.columns)
    assert list(saved_metrics.columns) == list(result.metrics.columns)
    assert saved_parameters["model"] == MODEL_NAME
