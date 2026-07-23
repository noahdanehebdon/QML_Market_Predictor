import pandas as pd
import pytest

from market_qml.reporting.backtest_charts import (
    FIGURE_FILENAMES,
    generate_backtest_charts,
    render_chart_report,
    save_chart_report,
)


def _portfolio_returns() -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(["logistic_regression", "qcnn"]):
        for index, date in enumerate(pd.date_range("2025-01-01", periods=30)):
            rows.append(
                {
                    "model_name": model,
                    "split_id": index // 10,
                    "date": date,
                    "net_return": 0.002 * (model_index + 1) + (index % 3 - 1) * 0.001,
                    "benchmark_return": 0.001 + (index % 2) * 0.0005,
                    "rebalance_frequency": 5,
                }
            )
    return pd.DataFrame(rows)


def _model_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": ["logistic_regression", "qcnn"],
            "classification_roc_auc": [0.57, 0.54],
            "ranking_rank_information_coefficient": [0.04, 0.06],
            "portfolio_net_sharpe": [1.2, 1.0],
        }
    )


def _regime_metrics() -> pd.DataFrame:
    rows = []
    for regime_type, regimes in {
        "volatility_regime": ["low", "high"],
        "rate_regime": ["rising", "falling"],
    }.items():
        for regime in regimes:
            for model, value in [("logistic_regression", 0.57), ("qcnn", 0.54)]:
                rows.append(
                    {
                        "regime_type": regime_type,
                        "regime": regime,
                        "model_name": model,
                        "meets_minimum_rows": True,
                        "roc_auc": value,
                    }
                )
    return pd.DataFrame(rows)


def test_generate_backtest_charts_creates_every_figure_and_report(tmp_path) -> None:
    figure_dir = tmp_path / "figures"
    paths = generate_backtest_charts(
        portfolio_returns=_portfolio_returns(),
        model_summary=_model_summary(),
        regime_metrics=_regime_metrics(),
        output_dir=figure_dir,
        rolling_window=5,
    )

    assert set(paths) == set(FIGURE_FILENAMES)
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 1_000

    report_path = tmp_path / "backtest_charts.md"
    markdown = render_chart_report(paths, report_path=report_path)
    save_chart_report(markdown, report_path)
    assert all(filename in markdown for filename in FIGURE_FILENAMES.values())
    assert report_path.read_text(encoding="utf-8") == markdown


@pytest.mark.parametrize("rolling_window", [0, 1])
def test_chart_generation_rejects_short_rolling_window(
    tmp_path, rolling_window
) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        generate_backtest_charts(
            portfolio_returns=_portfolio_returns(),
            model_summary=_model_summary(),
            regime_metrics=_regime_metrics(),
            output_dir=tmp_path,
            rolling_window=rolling_window,
        )


def test_chart_generation_rejects_missing_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        generate_backtest_charts(
            portfolio_returns=_portfolio_returns().drop(columns="net_return"),
            model_summary=_model_summary(),
            regime_metrics=_regime_metrics(),
            output_dir=tmp_path,
        )


def test_chart_generation_requires_rebalance_metadata_for_annualization(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="rebalance_frequency"):
        generate_backtest_charts(
            portfolio_returns=_portfolio_returns().drop(columns="rebalance_frequency"),
            model_summary=_model_summary(),
            regime_metrics=_regime_metrics(),
            output_dir=tmp_path,
        )
