import pandas as pd
import pytest
import yaml

from market_qml.backtest.portfolio import (
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_RETURN_HORIZON_DAYS,
    DEFAULT_TRANSACTION_COST_BPS,
    TRADING_DAYS_PER_YEAR,
    PORTFOLIO_RETURN_COLUMNS,
    PORTFOLIO_RISK_COLUMNS,
    load_prediction_tables,
    run_portfolio_backtest,
    save_portfolio_returns,
    save_portfolio_risk_metrics,
    summarize_portfolio_risk,
)
from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS


def test_backtest_config_matches_executable_defaults():
    with open("configs/backtest.yaml", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)["backtest"]

    assert config["return_horizon_days"] == DEFAULT_RETURN_HORIZON_DAYS
    assert config["rebalance_frequency"] == DEFAULT_REBALANCE_FREQUENCY
    assert config["trading_days_per_year"] == TRADING_DAYS_PER_YEAR
    assert config["transaction_cost_bps"] == DEFAULT_TRANSACTION_COST_BPS


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


def test_run_portfolio_backtest_selects_top_k_and_computes_gross_net_returns():
    result = run_portfolio_backtest(
        _predictions(),
        top_k=2,
        transaction_cost_bps=10,
        rebalance_frequency=1,
        return_horizon_days=1,
    )

    assert list(result.columns) == PORTFOLIO_RETURN_COLUMNS
    assert result["selected_count"].tolist() == [2, 2]
    assert result["turnover"].tolist() == pytest.approx([1.0, 0.5])
    assert result["transaction_cost"].tolist() == pytest.approx([0.001, 0.0005])
    assert result["gross_return"].tolist() == pytest.approx([0.035, 0.04])
    assert result["net_return"].tolist() == pytest.approx([0.034, 0.0395])
    assert result["benchmark_return"].tolist() == pytest.approx([0.01, 0.01])
    assert result["gross_excess_return"].tolist() == pytest.approx([0.025, 0.03])
    assert result["net_excess_return"].tolist() == pytest.approx([0.024, 0.0295])
    assert result["cumulative_gross_return"].iloc[-1] == pytest.approx(
        (1.035 * 1.04) - 1
    )
    assert result["cumulative_net_return"].iloc[-1] == pytest.approx(
        (1.034 * 1.0395) - 1
    )
    assert result["benchmark_cumulative_return"].iloc[-1] == pytest.approx(
        (1.01 * 1.01) - 1
    )


def test_run_portfolio_backtest_supports_top_fraction():
    result = run_portfolio_backtest(
        _predictions(),
        top_fraction=0.25,
        rebalance_frequency=1,
        return_horizon_days=1,
    )

    assert result["selected_count"].tolist() == [1, 1]
    assert result["gross_return"].tolist() == pytest.approx([0.04, 0.05])
    assert result["net_return"].tolist() == pytest.approx([0.039, 0.05])
    assert result["gross_excess_return"].tolist() == pytest.approx([0.03, 0.04])


def test_run_portfolio_backtest_supports_multiple_models():
    predictions = pd.concat(
        [_predictions("model_a"), _predictions("model_b")],
        ignore_index=True,
    )

    result = run_portfolio_backtest(
        predictions, top_k=1, rebalance_frequency=1, return_horizon_days=1
    )

    assert result["model_name"].tolist() == [
        "model_a",
        "model_a",
        "model_b",
        "model_b",
    ]


def test_summarize_portfolio_risk_reports_split_and_overall_metrics():
    returns = run_portfolio_backtest(
        _predictions(),
        top_k=2,
        transaction_cost_bps=10,
        rebalance_frequency=1,
        return_horizon_days=1,
    )
    risk = summarize_portfolio_risk(returns, periods_per_year=252)

    assert list(risk.columns) == PORTFOLIO_RISK_COLUMNS
    assert risk["scope"].tolist() == ["split", "overall"]
    assert risk["rows"].tolist() == [2, 2]
    assert risk["periods_per_year"].tolist() == [252, 252]
    assert risk["return_horizon_days"].tolist() == [1, 1]
    assert risk["transaction_cost_bps"].tolist() == [10, 10]
    assert risk.loc[0, "cumulative_gross_return"] == pytest.approx(
        returns["cumulative_gross_return"].iloc[-1]
    )
    assert risk.loc[0, "cumulative_net_return"] == pytest.approx(
        returns["cumulative_net_return"].iloc[-1]
    )
    assert risk.loc[0, "net_volatility"] >= 0
    assert risk.loc[0, "net_max_drawdown"] <= 0
    assert risk.loc[0, "hit_rate"] == pytest.approx(1.0)
    assert risk.loc[0, "excess_hit_rate"] == pytest.approx(1.0)
    assert risk.loc[0, "average_turnover"] == pytest.approx(0.75)
    assert risk.loc[0, "total_transaction_cost"] == pytest.approx(0.0015)
    assert pd.isna(risk.loc[1, "split_id"])


def test_run_portfolio_backtest_rejects_invalid_selection():
    with pytest.raises(ValueError, match="top_k"):
        run_portfolio_backtest(_predictions(), top_k=0)

    with pytest.raises(ValueError, match="transaction_cost_bps"):
        run_portfolio_backtest(_predictions(), transaction_cost_bps=-1)

    with pytest.raises(ValueError, match="rebalance_frequency"):
        run_portfolio_backtest(_predictions(), rebalance_frequency=0)

    with pytest.raises(ValueError, match="overlapping"):
        run_portfolio_backtest(
            _predictions(), rebalance_frequency=1, return_horizon_days=5
        )


def test_run_portfolio_backtest_defaults_to_horizon_rebalance_frequency():
    result = run_portfolio_backtest(_predictions(), top_k=2)

    assert result["rebalance_frequency"].tolist() == [5]
    assert result["date"].tolist() == [pd.Timestamp("2024-01-01")]
    assert result["selected_count"].tolist() == [2]


def test_five_day_returns_use_frequency_aware_annualization():
    returns = pd.DataFrame(
        {
            column: [value, value]
            for column, value in {
                "model_name": "model_a",
                "split_id": 0,
                "date": pd.Timestamp("2024-01-01"),
                "return_horizon_days": 5,
                "rebalance_frequency": 5,
                "transaction_cost_bps": 10.0,
                "selected_count": 1,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "gross_return": 0.0,
                "net_return": 0.0,
                "benchmark_return": 0.0,
                "gross_excess_return": 0.0,
                "net_excess_return": 0.0,
                "cumulative_gross_return": 0.0,
                "cumulative_net_return": 0.0,
                "benchmark_cumulative_return": 0.0,
                "cumulative_gross_excess_return": 0.0,
                "cumulative_net_excess_return": 0.0,
            }.items()
        },
        columns=PORTFOLIO_RETURN_COLUMNS,
    )
    returns["date"] = pd.to_datetime(["2024-01-01", "2024-01-08"])
    returns["net_return"] = [0.01, 0.03]
    returns["gross_return"] = returns["net_return"]
    risk = summarize_portfolio_risk(returns)

    expected_periods = 252 / 5
    expected_sharpe = returns["net_return"].mean() / returns["net_return"].std() * (
        expected_periods**0.5
    )
    assert risk.loc[0, "periods_per_year"] == pytest.approx(expected_periods)
    assert risk.loc[0, "net_sharpe"] == pytest.approx(expected_sharpe)


def test_load_prediction_tables_and_save_portfolio_returns(tmp_path):
    prediction_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "portfolio.parquet"
    risk_output_path = tmp_path / "portfolio_risk.parquet"
    _predictions().to_parquet(prediction_path, index=False)

    predictions = load_prediction_tables([prediction_path])
    result = run_portfolio_backtest(
        predictions, top_k=2, rebalance_frequency=1, return_horizon_days=1
    )
    risk = summarize_portfolio_risk(result)
    save_portfolio_returns(result, output_path)
    save_portfolio_risk_metrics(risk, risk_output_path)
    saved = pd.read_parquet(output_path)
    saved_risk = pd.read_parquet(risk_output_path)

    assert len(predictions) == 8
    assert output_path.exists()
    assert risk_output_path.exists()
    assert list(saved.columns) == PORTFOLIO_RETURN_COLUMNS
    assert list(saved_risk.columns) == PORTFOLIO_RISK_COLUMNS
