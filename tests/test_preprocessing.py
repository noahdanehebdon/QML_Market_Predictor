import warnings

import numpy as np
import pandas as pd
import pytest

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets
from market_qml.models.preprocessing import (
    fit_preprocessor,
    fit_transform_train_validation,
    load_preprocessor,
    preprocess_dataset,
    save_preprocessor,
    transform_features,
)


def _dataset(X: pd.DataFrame) -> ModelingDataset:
    return ModelingDataset(
        X=X,
        y=pd.Series([1] * len(X), name="outperform_spy_5d"),
        metadata=pd.DataFrame(
            {
                "symbol": ["AAPL"] * len(X),
                "date": pd.date_range("2024-01-01", periods=len(X), freq="D"),
            }
        ),
    )


def test_fit_preprocessor_uses_training_data_only_for_statistics():
    train = _dataset(
        pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, None],
                "feature_b": [10.0, 10.0, 10.0],
            }
        )
    )
    validation = _dataset(
        pd.DataFrame(
            {
                "feature_a": [9999.0, None],
                "feature_b": [-9999.0, 9999.0],
            }
        )
    )

    result = fit_transform_train_validation(
        TrainValidationDatasets(train=train, validation=validation)
    )

    assert result.preprocessor.fill_values["feature_a"] == pytest.approx(1.5)
    assert result.preprocessor.means["feature_a"] == pytest.approx(1.5)
    assert result.preprocessor.scales["feature_b"] == pytest.approx(1.0)
    assert result.validation.X.loc[0, "feature_a"] > 1000
    assert result.validation.X.loc[1, "feature_a"] == pytest.approx(0.0)


def test_transform_features_reuses_train_fitted_preprocessor():
    train_X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    preprocessor = fit_preprocessor(train_X)

    result = transform_features(pd.DataFrame({"x": [2.0, None]}), preprocessor)

    assert result.loc[0, "x"] == pytest.approx(0.0)
    assert result.loc[1, "x"] == pytest.approx(0.0)


def test_fit_preprocessor_handles_all_missing_and_boolean_features():
    train_X = pd.DataFrame(
        {
            "all_missing": [None, None, None],
            "flag": [True, False, True],
        }
    )

    preprocessor = fit_preprocessor(train_X)
    result = transform_features(train_X, preprocessor)

    assert preprocessor.fill_values["all_missing"] == pytest.approx(0.0)
    assert result["all_missing"].tolist() == [0.0, 0.0, 0.0]
    assert "flag" in result.columns


def test_preprocessor_treats_infinity_as_missing_and_returns_finite_matrices():
    train = _dataset(
        pd.DataFrame(
            {
                "mixed": [1.0, np.inf, -np.inf],
                "all_non_finite": [np.inf, -np.inf, np.nan],
            }
        )
    )
    validation = _dataset(
        pd.DataFrame(
            {
                "mixed": [np.inf, 2.0],
                "all_non_finite": [-np.inf, np.nan],
            }
        )
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = fit_transform_train_validation(
            TrainValidationDatasets(train=train, validation=validation)
        )

    assert not [
        warning
        for warning in captured
        if "invalid value encountered" in str(warning.message)
    ]
    assert np.isfinite(result.train.X.to_numpy()).all()
    assert np.isfinite(result.validation.X.to_numpy()).all()
    assert result.preprocessor.fill_values["mixed"] == pytest.approx(1.0)
    assert result.preprocessor.fill_values["all_non_finite"] == pytest.approx(0.0)


def test_preprocess_dataset_preserves_target_and_metadata():
    dataset = _dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    preprocessor = fit_preprocessor(dataset.X)

    result = preprocess_dataset(dataset, preprocessor)

    assert list(result.X.columns) == ["x"]
    assert result.y.equals(dataset.y)
    assert result.metadata.equals(dataset.metadata)


def test_preprocessor_can_be_saved_and_reused(tmp_path):
    path = tmp_path / "preprocessor.pkl"
    preprocessor = fit_preprocessor(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    save_preprocessor(preprocessor, path)
    loaded = load_preprocessor(path)
    result = transform_features(pd.DataFrame({"x": [2.0]}), loaded)

    assert path.exists()
    assert loaded.feature_columns == ["x"]
    assert result.loc[0, "x"] == pytest.approx(0.0)


def test_transform_features_requires_fitted_columns():
    preprocessor = fit_preprocessor(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))

    with pytest.raises(ValueError, match="missing fitted columns"):
        transform_features(pd.DataFrame({"other": [1.0]}), preprocessor)


def test_fit_preprocessor_rejects_empty_training_matrix():
    with pytest.raises(ValueError, match="empty training matrix"):
        fit_preprocessor(pd.DataFrame())
