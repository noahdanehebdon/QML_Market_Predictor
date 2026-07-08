import pandas as pd
import pytest

from market_qml.backtest.portfolio import (
    PORTFOLIO_RETURN_COLUMNS,
    load_prediction_tables,
    run_portfolio_backtest,
    save_portfolio_returns,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


def _predictions(model_name: str = "model_a") -> pd.DataFrame:
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
            "split_id": [0] * 8,
        },
        columns=REQUIRED_PREDICTION_COLUMNS,
    )


def test_run_portfolio_backtest_selects_top_k_and_computes_cumulative_returns():
    result = run_portfolio_backtest(_predictions(), top_k=2)

    assert list(result.columns) == PORTFOLIO_RETURN_COLUMNS
    assert result["selected_count"].tolist() == [2, 2]
    assert result["portfolio_return"].tolist() == pytest.approx([0.035, 0.04])
    assert result["benchmark_return"].tolist() == pytest.approx([0.01, 0.01])
    assert result["excess_return"].tolist() == pytest.approx([0.025, 0.03])
    assert result["cumulative_return"].iloc[-1] == pytest.approx(
        (1.035 * 1.04) - 1
    )
    assert result["benchmark_cumulative_return"].iloc[-1] == pytest.approx(
        (1.01 * 1.01) - 1
    )


def test_run_portfolio_backtest_supports_top_fraction():
    result = run_portfolio_backtest(_predictions(), top_fraction=0.25)

    assert result["selected_count"].tolist() == [1, 1]
    assert result["portfolio_return"].tolist() == pytest.approx([0.04, 0.05])
    assert result["excess_return"].tolist() == pytest.approx([0.03, 0.04])


def test_run_portfolio_backtest_supports_multiple_models():
    predictions = pd.concat(
        [_predictions("model_a"), _predictions("model_b")],
        ignore_index=True,
    )

    result = run_portfolio_backtest(predictions, top_k=1)

    assert result["model_name"].tolist() == [
        "model_a",
        "model_a",
        "model_b",
        "model_b",
    ]


def test_run_portfolio_backtest_rejects_invalid_selection():
    with pytest.raises(ValueError, match="top_k"):
        run_portfolio_backtest(_predictions(), top_k=0)


def test_load_prediction_tables_and_save_portfolio_returns(tmp_path):
    prediction_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "portfolio.parquet"
    _predictions().to_parquet(prediction_path, index=False)

    predictions = load_prediction_tables([prediction_path])
    result = run_portfolio_backtest(predictions, top_k=2)
    save_portfolio_returns(result, output_path)
    saved = pd.read_parquet(output_path)

    assert len(predictions) == 8
    assert output_path.exists()
    assert list(saved.columns) == PORTFOLIO_RETURN_COLUMNS
