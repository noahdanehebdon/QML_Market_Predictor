import pandas as pd
import pytest

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.qml.interface import (
    BaseQMLModel,
    QMLModelConfig,
    build_qml_train_validation,
)


class DemoQMLModel(BaseQMLModel):
    def fit(self, dataset):
        self.fit_shape = dataset.X.shape
        return self

    def predict_scores(self, dataset):
        return [0.7 if value > 0 else 0.3 for value in dataset.X["pca_00"]]


def _qml_sample() -> pd.DataFrame:
    rows = []
    for role, dates in [
        ("train", pd.date_range("2024-01-01", periods=2, freq="D")),
        ("validation", pd.date_range("2024-02-01", periods=2, freq="D")),
    ]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "symbol": "AAPL" if index == 0 else "MSFT",
                    "date": date,
                    "split_id": 1,
                    "sample_role": role,
                    "target": index,
                    "forward_return_5d": 0.01 + index,
                    "forward_excess_return_5d": 0.02 + index,
                    "pca_00": float(index),
                    "macro_pca_00": float(index + 10),
                }
            )
    return pd.DataFrame(rows)


def test_build_qml_train_validation_defines_expected_input_format():
    data = build_qml_train_validation(_qml_sample(), split_id=1)

    assert data.split_id == 1
    assert data.feature_columns == ["macro_pca_00", "pca_00"]
    assert list(data.train.X.columns) == data.feature_columns
    assert data.train.y.tolist() == [0, 1]
    assert data.validation.metadata["symbol"].tolist() == ["AAPL", "MSFT"]
    assert "forward_return_5d" in data.validation.metadata.columns


def test_qml_model_outputs_standard_prediction_table_with_config_and_seed():
    data = build_qml_train_validation(_qml_sample(), split_id=1)
    model = DemoQMLModel(
        QMLModelConfig(
            model_name="demo_qml",
            seed=123,
            params={"layers": 2},
        )
    )

    result = model.train(data)

    assert model.model_name == "demo_qml"
    assert model.seed == 123
    assert model.config.params == {"layers": 2}
    assert model.fit_shape == (2, 2)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == ["demo_qml"]
    assert result.predictions["split_id"].unique().tolist() == [1]
    assert result.predictions["y_score"].tolist() == [0.3, 0.7]


def test_build_qml_train_validation_validates_sample_shape():
    with pytest.raises(ValueError, match="missing required columns"):
        build_qml_train_validation(pd.DataFrame({"symbol": ["AAPL"]}))

    with pytest.raises(ValueError, match="PCA component"):
        build_qml_train_validation(
            _qml_sample().drop(columns=["pca_00", "macro_pca_00"]),
            split_id=1,
        )

    with pytest.raises(ValueError, match="validation"):
        build_qml_train_validation(
            _qml_sample()[lambda frame: frame["sample_role"] == "train"],
            split_id=1,
        )


def test_qml_model_rejects_wrong_score_count():
    class BadScoreModel(DemoQMLModel):
        def predict_scores(self, dataset):
            return [0.1]

    data = build_qml_train_validation(_qml_sample(), split_id=1)

    with pytest.raises(ValueError, match="one score per validation row"):
        BadScoreModel(QMLModelConfig(model_name="bad_qml")).train(data)
