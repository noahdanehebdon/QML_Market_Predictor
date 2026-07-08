import pandas as pd

from market_qml.utils.mlflow_tracking import log_walk_forward_backtest_run


class _FakeRun:
    def __init__(self):
        self.info = type("Info", (), {"run_id": "run-123"})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflow:
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None
        self.params = {}
        self.metrics = {}
        self.artifacts = []

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None):
        self.run_name = run_name
        return _FakeRun()

    def log_params(self, params):
        self.params.update(params)

    def log_metric(self, name, value):
        self.metrics[name] = value

    def log_artifact(self, path):
        self.artifacts.append(path)


def test_log_walk_forward_backtest_run_logs_params_metrics_and_artifacts(tmp_path):
    fake_mlflow = _FakeMlflow()
    output_path = tmp_path / "predictions.parquet"
    output_path.write_text("placeholder", encoding="utf-8")

    predictions = pd.DataFrame(
        {
            "model_name": ["logistic_regression", "logistic_regression"],
            "split_id": [0, 0],
        }
    )
    splits = pd.DataFrame(
        {
            "split_id": [0],
            "train_start_date": [pd.Timestamp("2024-01-01")],
            "validation_end_date": [pd.Timestamp("2024-02-01")],
        }
    )
    features = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2024-01-01")],
            "momentum": [1.0],
        }
    )
    labels = pd.DataFrame({"label_horizon_days": [5]})
    classification = pd.DataFrame(
        {
            "model_name": ["logistic_regression"],
            "scope": ["overall"],
            "split_id": [pd.NA],
            "roc_auc": [0.7],
        }
    )
    ranking = pd.DataFrame(
        {
            "model_name": ["logistic_regression"],
            "scope": ["overall"],
            "split_id": [pd.NA],
            "long_short_spread": [0.02],
        }
    )
    risk = pd.DataFrame(
        {
            "model_name": ["logistic_regression"],
            "scope": ["overall"],
            "split_id": [pd.NA],
            "net_sharpe": [1.5],
        }
    )

    run_id = log_walk_forward_backtest_run(
        output_paths={"predictions": output_path},
        predictions=predictions,
        splits=splits,
        features=features,
        labels=labels,
        classification_metrics=classification,
        ranking_metrics=ranking,
        portfolio_risk_metrics=risk,
        model_names=["logistic_regression"],
        top_k=None,
        top_fraction=0.1,
        transaction_cost_bps=10,
        rebalance_frequency=5,
        periods_per_year=252,
        max_splits=1,
        experiment_name="test-experiment",
        run_name="test-run",
        tracking_uri="file:mlruns",
        mlflow_module=fake_mlflow,
    )

    assert run_id == "run-123"
    assert fake_mlflow.tracking_uri == "file:mlruns"
    assert fake_mlflow.experiment_name == "test-experiment"
    assert fake_mlflow.run_name == "test-run"
    assert fake_mlflow.params["model_names"] == "logistic_regression"
    assert fake_mlflow.params["feature_count"] == 1
    assert fake_mlflow.params["feature_columns"] == "momentum"
    assert fake_mlflow.params["target_horizon_days"] == 5
    assert fake_mlflow.params["rebalance_frequency"] == 5
    assert "git_commit" in fake_mlflow.params
    assert fake_mlflow.metrics["classification.logistic_regression.roc_auc"] == 0.7
    assert fake_mlflow.metrics["ranking.logistic_regression.long_short_spread"] == 0.02
    assert fake_mlflow.metrics["portfolio.logistic_regression.net_sharpe"] == 1.5
    assert fake_mlflow.artifacts == [str(output_path)]
