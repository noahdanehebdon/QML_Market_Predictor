import pandas as pd

import scripts.run_walk_forward_backtest as runner
from scripts.run_walk_forward_backtest import run_walk_forward_backtest


def _features() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    for symbol, offset in [("AAPL", 1.0), ("MSFT", 2.0), ("NVDA", 3.0)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": 100 + offset + index,
                    "momentum": offset * (index + 1),
                    "volatility": 0.1 * offset,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    rows = []
    label_values = {
        "AAPL": [0, 1, 1, 0],
        "MSFT": [1, 0, 1, 0],
        "NVDA": [0, 1, 0, 1],
    }
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    for symbol in ["AAPL", "MSFT", "NVDA"]:
        for index, date in enumerate(dates):
            outperform = label_values[symbol][index]
            forward_excess = 0.02 if outperform else -0.01
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "label_horizon_days": 5,
                    "forward_return_5d": 0.01 + forward_excess,
                    "spy_forward_return_5d": 0.01,
                    "forward_excess_return_5d": forward_excess,
                    "outperform_spy_5d": outperform,
                }
            )
    return pd.DataFrame(rows)


def _splits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_id": [0],
            "train_start_date": [pd.Timestamp("2024-01-01")],
            "train_end_date": [pd.Timestamp("2024-01-02")],
            "validation_start_date": [pd.Timestamp("2024-01-03")],
            "validation_end_date": [pd.Timestamp("2024-01-04")],
            "train_days": [2],
            "validation_days": [2],
            "train_rows": [6],
            "validation_rows": [6],
        }
    )


def test_run_walk_forward_backtest_writes_report_bundle(tmp_path):
    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        model_names=["logistic_regression"],
        output_dir=tmp_path,
        transaction_cost_bps=10,
    )

    assert sorted(outputs) == [
        "classification_metrics",
        "portfolio_backtest",
        "portfolio_risk_metrics",
        "predictions",
        "ranking_metrics",
    ]
    for path in outputs.values():
        assert path.exists()

    predictions = pd.read_parquet(outputs["predictions"])
    classification = pd.read_parquet(outputs["classification_metrics"])
    ranking = pd.read_parquet(outputs["ranking_metrics"])
    portfolio = pd.read_parquet(outputs["portfolio_backtest"])
    risk = pd.read_parquet(outputs["portfolio_risk_metrics"])

    assert predictions["model_name"].unique().tolist() == ["logistic_regression"]
    assert predictions["split_id"].unique().tolist() == [0]
    assert len(predictions) == 6
    assert classification["scope"].tolist() == ["split", "overall"]
    assert set(ranking["scope"]) == {"date", "split", "overall"}
    assert "net_return" in portfolio.columns
    assert "net_sharpe" in risk.columns


def test_run_walk_forward_backtest_respects_max_splits(tmp_path):
    splits = pd.concat([_splits(), _splits().assign(split_id=1)], ignore_index=True)

    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=splits,
        model_names=["logistic_regression"],
        output_dir=tmp_path,
        max_splits=1,
    )

    predictions = pd.read_parquet(outputs["predictions"])

    assert predictions["split_id"].unique().tolist() == [0]


def test_run_walk_forward_backtest_supports_huber_regression_lane(tmp_path):
    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        model_names=["huber_regression"],
        output_dir=tmp_path,
        transaction_cost_bps=10,
    )

    predictions = pd.read_parquet(outputs["predictions"])
    classification = pd.read_parquet(outputs["classification_metrics"])
    ranking = pd.read_parquet(outputs["ranking_metrics"])
    portfolio = pd.read_parquet(outputs["portfolio_backtest"])
    risk = pd.read_parquet(outputs["portfolio_risk_metrics"])

    assert predictions["model_name"].unique().tolist() == ["huber_regression"]
    assert predictions["y_true"].dtype.kind == "f"
    assert classification.empty
    assert set(ranking["model_name"]) == {"huber_regression"}
    assert set(portfolio["model_name"]) == {"huber_regression"}
    assert set(risk["model_name"]) == {"huber_regression"}


def test_run_walk_forward_backtest_supports_elastic_net_regression_lane(tmp_path):
    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        model_names=["elastic_net"],
        output_dir=tmp_path,
        transaction_cost_bps=10,
    )

    predictions = pd.read_parquet(outputs["predictions"])
    classification = pd.read_parquet(outputs["classification_metrics"])
    ranking = pd.read_parquet(outputs["ranking_metrics"])
    portfolio = pd.read_parquet(outputs["portfolio_backtest"])
    risk = pd.read_parquet(outputs["portfolio_risk_metrics"])

    assert predictions["model_name"].unique().tolist() == ["elastic_net"]
    assert predictions["y_true"].dtype.kind == "f"
    assert classification.empty
    assert set(ranking["model_name"]) == {"elastic_net"}
    assert set(portfolio["model_name"]) == {"elastic_net"}
    assert set(risk["model_name"]) == {"elastic_net"}


def test_run_walk_forward_backtest_supports_random_forest_regressor_lane(tmp_path):
    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        model_names=["random_forest_regressor"],
        output_dir=tmp_path,
        transaction_cost_bps=10,
    )

    predictions = pd.read_parquet(outputs["predictions"])
    classification = pd.read_parquet(outputs["classification_metrics"])
    ranking = pd.read_parquet(outputs["ranking_metrics"])
    portfolio = pd.read_parquet(outputs["portfolio_backtest"])
    risk = pd.read_parquet(outputs["portfolio_risk_metrics"])

    assert predictions["model_name"].unique().tolist() == ["random_forest_regressor"]
    assert predictions["y_true"].dtype.kind == "f"
    assert classification.empty
    assert set(ranking["model_name"]) == {"random_forest_regressor"}
    assert set(portfolio["model_name"]) == {"random_forest_regressor"}
    assert set(risk["model_name"]) == {"random_forest_regressor"}


def test_run_walk_forward_backtest_can_log_mlflow_run(tmp_path, monkeypatch):
    calls = []

    def fake_log_walk_forward_backtest_run(**kwargs):
        calls.append(kwargs)
        return "run-123"

    monkeypatch.setattr(
        runner,
        "log_walk_forward_backtest_run",
        fake_log_walk_forward_backtest_run,
    )

    outputs = run_walk_forward_backtest(
        features=_features(),
        labels=_labels(),
        splits=_splits(),
        model_names=["logistic_regression"],
        output_dir=tmp_path,
        enable_mlflow=True,
        mlflow_experiment="test-experiment",
        mlflow_run_name="test-run",
        mlflow_tracking_uri="file:mlruns",
    )

    assert len(calls) == 1
    assert calls[0]["output_paths"] == outputs
    assert calls[0]["model_names"] == ["logistic_regression"]
    assert calls[0]["experiment_name"] == "test-experiment"
    assert calls[0]["run_name"] == "test-run"
    assert calls[0]["tracking_uri"] == "file:mlruns"
