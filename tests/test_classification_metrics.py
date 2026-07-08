import pandas as pd
import pytest

from market_qml.backtest.classification_metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    evaluate_classification_metrics,
    load_prediction_tables,
    save_classification_metrics,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


def _prediction_table(
    *,
    model_name: str = "demo_model",
    split_id: int = 0,
    y_true: list[float] | None = None,
    y_score: list[float] | None = None,
) -> pd.DataFrame:
    y_true = [0, 1, 1, 0] if y_true is None else y_true
    y_score = [0.1, 0.8, 0.6, 0.4] if y_score is None else y_score
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA", "AMZN"][: len(y_true)],
            "date": pd.date_range("2024-01-01", periods=len(y_true), freq="D"),
            "y_true": y_true,
            "y_score": y_score,
            "forward_return": [0.01, 0.02, 0.03, -0.01][: len(y_true)],
            "forward_excess_return": [0.0, 0.01, 0.02, -0.02][: len(y_true)],
            "model_name": [model_name] * len(y_true),
            "split_id": [split_id] * len(y_true),
        },
        columns=REQUIRED_PREDICTION_COLUMNS,
    )


def test_evaluate_classification_metrics_reports_by_split_and_overall():
    predictions = pd.concat(
        [
            _prediction_table(split_id=0),
            _prediction_table(split_id=1, y_score=[0.2, 0.7, 0.9, 0.3]),
        ],
        ignore_index=True,
    )

    metrics = evaluate_classification_metrics(predictions)

    assert list(metrics.columns) == CLASSIFICATION_METRIC_COLUMNS
    assert metrics["scope"].tolist() == ["split", "split", "overall"]
    assert metrics["split_id"].tolist()[:2] == [0, 1]
    assert pd.isna(metrics.loc[2, "split_id"])
    assert metrics["rows"].tolist() == [4, 4, 8]
    assert metrics["positive_labels"].tolist() == [2, 2, 4]
    assert metrics["accuracy"].tolist() == [1.0, 1.0, 1.0]
    assert metrics["precision"].tolist() == [1.0, 1.0, 1.0]
    assert metrics["recall"].tolist() == [1.0, 1.0, 1.0]
    assert metrics["roc_auc"].tolist() == [1.0, 1.0, 1.0]
    assert metrics["brier_score"].between(0, 1).all()


def test_evaluate_classification_metrics_supports_multiple_models():
    predictions = pd.concat(
        [
            _prediction_table(model_name="model_a"),
            _prediction_table(model_name="model_b", y_score=[0.4, 0.6, 0.7, 0.2]),
        ],
        ignore_index=True,
    )

    metrics = evaluate_classification_metrics(predictions)

    assert metrics["model_name"].tolist() == [
        "model_a",
        "model_b",
        "model_a",
        "model_b",
    ]
    assert metrics["scope"].tolist() == ["split", "split", "overall", "overall"]


def test_evaluate_classification_metrics_rejects_non_binary_targets():
    predictions = _prediction_table(y_true=[-0.02, 0.01, 0.03, -0.01])

    with pytest.raises(ValueError, match="binary y_true"):
        evaluate_classification_metrics(predictions)


def test_load_prediction_tables_skips_non_binary_tables(tmp_path):
    binary_path = tmp_path / "predictions_logistic.parquet"
    regression_path = tmp_path / "predictions_ridge.parquet"
    _prediction_table(model_name="logistic").to_parquet(binary_path, index=False)
    _prediction_table(
        model_name="ridge",
        y_true=[-0.02, 0.01, 0.03, -0.01],
    ).to_parquet(regression_path, index=False)

    predictions = load_prediction_tables([binary_path, regression_path])

    assert predictions["model_name"].unique().tolist() == ["logistic"]


def test_classification_metrics_can_be_saved(tmp_path):
    metrics = evaluate_classification_metrics(_prediction_table())
    metrics_path = tmp_path / "classification_metrics.parquet"

    save_classification_metrics(metrics, metrics_path)
    saved = pd.read_parquet(metrics_path)

    assert metrics_path.exists()
    assert list(saved.columns) == CLASSIFICATION_METRIC_COLUMNS
