import pandas as pd
import pytest

from market_qml.backtest.ranking_metrics import (
    RANKING_METRIC_COLUMNS,
    evaluate_ranking_metrics,
    load_prediction_tables,
    save_ranking_metrics,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


def _predictions(model_name: str = "model_a", split_id: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
            "y_true": [1, 1, 0, 0, 1, 0, 1, 0],
            "y_score": [0.9, 0.7, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4],
            "forward_return": [0.04, 0.03, -0.01, -0.02, 0.05, -0.02, 0.03, 0.0],
            "forward_excess_return": [
                0.03,
                0.02,
                -0.02,
                -0.03,
                0.04,
                -0.03,
                0.02,
                -0.01,
            ],
            "model_name": [model_name] * 8,
            "split_id": [split_id] * 8,
        },
        columns=REQUIRED_PREDICTION_COLUMNS,
    )


def test_evaluate_ranking_metrics_reports_date_split_and_overall_rows():
    metrics = evaluate_ranking_metrics(_predictions(), top_fraction=0.25)

    assert list(metrics.columns) == RANKING_METRIC_COLUMNS
    assert metrics["scope"].tolist() == ["date", "date", "split", "overall"]
    assert metrics["rows"].tolist() == [4, 4, 8, 8]
    assert metrics.loc[0, "information_coefficient"] > 0.99
    assert metrics.loc[0, "rank_information_coefficient"] == pytest.approx(1.0)
    assert metrics.loc[0, "top_decile_return"] == pytest.approx(0.03)
    assert metrics.loc[0, "bottom_decile_return"] == pytest.approx(-0.03)
    assert metrics.loc[0, "long_short_spread"] == pytest.approx(0.06)
    assert pd.isna(metrics.loc[3, "split_id"])
    assert pd.isna(metrics.loc[3, "date"])


def test_evaluate_ranking_metrics_supports_multiple_models():
    predictions = pd.concat(
        [
            _predictions("model_a", split_id=0),
            _predictions("model_b", split_id=0).assign(
                y_score=[0.1, 0.3, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6]
            ),
        ],
        ignore_index=True,
    )

    metrics = evaluate_ranking_metrics(predictions, top_fraction=0.25)
    overall = metrics[metrics["scope"] == "overall"].sort_values("model_name")

    assert overall["model_name"].tolist() == ["model_a", "model_b"]
    assert overall.iloc[0]["long_short_spread"] > overall.iloc[1]["long_short_spread"]


def test_evaluate_ranking_metrics_rejects_bad_top_fraction():
    with pytest.raises(ValueError, match="top_fraction"):
        evaluate_ranking_metrics(_predictions(), top_fraction=0.8)


def test_load_prediction_tables_and_save_metrics(tmp_path):
    prediction_path = tmp_path / "predictions.parquet"
    metrics_path = tmp_path / "ranking_metrics.parquet"
    _predictions().to_parquet(prediction_path, index=False)

    predictions = load_prediction_tables([prediction_path])
    metrics = evaluate_ranking_metrics(predictions, top_fraction=0.25)
    save_ranking_metrics(metrics, metrics_path)
    saved = pd.read_parquet(metrics_path)

    assert len(predictions) == 8
    assert metrics_path.exists()
    assert list(saved.columns) == RANKING_METRIC_COLUMNS
