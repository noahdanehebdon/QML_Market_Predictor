import pandas as pd
import pytest

from market_qml.reporting.dashboard import (
    latest_signal_report,
    portfolio_series,
    qml_experiment_summary,
    top_ranked_stocks,
)


def test_dashboard_loads_signals_and_excludes_benchmark(tmp_path):
    signals = pd.DataFrame(
        {
            "date": ["2025-01-02"] * 3,
            "rank": [2, 1, 3],
            "symbol": ["AAPL", "SPY", "MSFT"],
            "predicted_outperformance_probability": [0.7, 0.8, 0.6],
            "is_benchmark": [False, True, False],
        }
    )
    signals.to_csv(tmp_path / "daily_signal.csv", index=False)

    loaded = latest_signal_report(tmp_path)
    ranked = top_ranked_stocks(tmp_path, limit=1)

    assert loaded["rank"].tolist() == [1, 2, 3]
    assert ranked["symbol"].tolist() == ["AAPL"]


def test_dashboard_builds_cumulative_returns_and_drawdowns(tmp_path):
    output = tmp_path / "qml_comparison"
    output.mkdir()
    pd.DataFrame(
        {
            "model_name": ["vqc", "vqc"],
            "date": ["2025-01-01", "2025-01-02"],
            "net_return": [0.10, -0.20],
        }
    ).to_parquet(output / "portfolio_returns.parquet", index=False)

    result = portfolio_series(tmp_path)

    assert result["cumulative_net_return"].tolist() == pytest.approx([0.10, -0.12])
    assert result["drawdown"].tolist() == pytest.approx([0.0, -0.20])


def test_dashboard_pivots_qml_metrics(tmp_path):
    output = tmp_path / "qml_comparison"
    output.mkdir()
    pd.DataFrame(
        {
            "model_name": ["vqc", "vqc", "qcnn"],
            "metric": ["roc_auc", "accuracy", "roc_auc"],
            "mean": [0.55, 0.52, 0.57],
        }
    ).to_parquet(output / "aggregate_metrics.parquet", index=False)

    result = qml_experiment_summary(tmp_path).set_index("model_name")

    assert result.loc["vqc", "roc_auc"] == pytest.approx(0.55)
    assert result.loc["qcnn", "roc_auc"] == pytest.approx(0.57)


def test_dashboard_rejects_non_positive_stock_limit(tmp_path):
    with pytest.raises(ValueError, match="limit must be positive"):
        top_ranked_stocks(tmp_path, limit=0)
