import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.models.random_forest import (
    MODEL_NAME,
    save_feature_importance,
    save_predictions,
    save_random_forest_model,
    train_random_forest,
)


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
                "forward_return_5d": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04][: len(X)],
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


def test_train_random_forest_outputs_required_prediction_columns():
    result = train_random_forest(
        _preprocessed(),
        n_estimators=20,
        min_samples_leaf=1,
    )

    assert isinstance(result.model, RandomForestClassifier)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["split_id"].unique().tolist() == [0]
    assert result.predictions["y_true"].tolist() == [0, 1]
    assert result.predictions["y_score"].between(0, 1).all()


def test_train_random_forest_outputs_feature_importance():
    data = _preprocessed()
    result = train_random_forest(data, n_estimators=20, min_samples_leaf=1)

    assert list(result.feature_importance.columns) == [
        "feature",
        "importance",
        "model",
        "rank",
    ]
    assert set(result.feature_importance["feature"]) == set(data.train.X.columns)
    assert result.feature_importance["model"].unique().tolist() == [MODEL_NAME]
    assert result.feature_importance["importance"].between(0, 1).all()
    assert result.feature_importance["rank"].tolist() == [1.0, 2.0, 3.0]


def test_train_random_forest_rejects_single_class_training_labels():
    data = _preprocessed()
    single_class_train = ModelingDataset(
        X=data.train.X,
        y=pd.Series([1] * len(data.train.y)),
        metadata=data.train.metadata,
    )

    with pytest.raises(ValueError, match="at least two classes"):
        train_random_forest(
            TrainValidationDatasets(
                train=single_class_train,
                validation=data.validation,
            )
        )


def test_random_forest_outputs_can_be_saved(tmp_path):
    result = train_random_forest(
        _preprocessed(),
        n_estimators=20,
        min_samples_leaf=1,
    )
    model_path = tmp_path / "model.pkl"
    prediction_path = tmp_path / "predictions.parquet"
    importance_path = tmp_path / "feature_importance.parquet"

    save_random_forest_model(result.model, model_path)
    save_predictions(result.predictions, prediction_path)
    save_feature_importance(result.feature_importance, importance_path)
    saved_predictions = pd.read_parquet(prediction_path)
    saved_importance = pd.read_parquet(importance_path)

    assert model_path.exists()
    assert prediction_path.exists()
    assert importance_path.exists()
    assert list(saved_predictions.columns) == list(result.predictions.columns)
    assert list(saved_importance.columns) == list(result.feature_importance.columns)
