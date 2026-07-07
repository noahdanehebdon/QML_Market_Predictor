import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.logistic_regression import (
    MODEL_NAME,
    save_logistic_regression_model,
    save_predictions,
    train_logistic_regression,
)
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
                "symbol": ["AAPL", "MSFT", "NVDA", "AMZN"][: len(X)],
                "date": pd.date_range(start, periods=len(X), freq="D"),
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
        [0, 0, 1, 1],
        start="2024-01-01",
    )
    validation = _dataset(
        pd.DataFrame(
            {
                "momentum": [-1.5, 1.5],
                "volatility": [0.45, 0.25],
            }
        ),
        [0, 1],
        start="2024-02-01",
    )
    return fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )


def test_train_logistic_regression_outputs_required_prediction_columns():
    result = train_logistic_regression(_preprocessed())

    assert isinstance(result.model, LogisticRegression)
    assert list(result.predictions.columns) == [
        "symbol",
        "date",
        "y_true",
        "y_score",
        "model",
    ]
    assert result.predictions["model"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["y_true"].tolist() == [0, 1]
    assert result.predictions["y_score"].between(0, 1).all()


def test_train_logistic_regression_rejects_single_class_training_labels():
    data = _preprocessed()
    single_class_train = ModelingDataset(
        X=data.train.X,
        y=pd.Series([1] * len(data.train.y)),
        metadata=data.train.metadata,
    )

    with pytest.raises(ValueError, match="at least two classes"):
        train_logistic_regression(
            TrainValidationDatasets(
                train=single_class_train,
                validation=data.validation,
            )
        )


def test_logistic_regression_outputs_can_be_saved(tmp_path):
    result = train_logistic_regression(_preprocessed())
    model_path = tmp_path / "model.pkl"
    prediction_path = tmp_path / "predictions.parquet"

    save_logistic_regression_model(result.model, model_path)
    save_predictions(result.predictions, prediction_path)
    saved_predictions = pd.read_parquet(prediction_path)

    assert model_path.exists()
    assert prediction_path.exists()
    assert list(saved_predictions.columns) == list(result.predictions.columns)
