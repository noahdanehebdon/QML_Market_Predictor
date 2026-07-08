"""MLflow experiment tracking helpers."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any

import pandas as pd


DEFAULT_EXPERIMENT_NAME = "QML Market Predictor"


def log_walk_forward_backtest_run(
    *,
    output_paths: dict[str, Path],
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    classification_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    portfolio_risk_metrics: pd.DataFrame,
    model_names: list[str],
    top_k: int | None,
    top_fraction: float,
    transaction_cost_bps: float,
    rebalance_frequency: int,
    periods_per_year: int,
    max_splits: int | None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    mlflow_module: Any | None = None,
) -> str:
    """Log a walk-forward backtest run to MLflow and return the run id."""
    mlflow = mlflow_module or _import_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as active_run:
        _log_params(
            mlflow=mlflow,
            predictions=predictions,
            splits=splits,
            features=features,
            labels=labels,
            model_names=model_names,
            top_k=top_k,
            top_fraction=top_fraction,
            transaction_cost_bps=transaction_cost_bps,
            rebalance_frequency=rebalance_frequency,
            periods_per_year=periods_per_year,
            max_splits=max_splits,
        )
        _log_overall_metrics(
            mlflow=mlflow,
            classification_metrics=classification_metrics,
            ranking_metrics=ranking_metrics,
            portfolio_risk_metrics=portfolio_risk_metrics,
        )
        for path in output_paths.values():
            mlflow.log_artifact(str(path))

        return active_run.info.run_id


def _log_params(
    *,
    mlflow,
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_names: list[str],
    top_k: int | None,
    top_fraction: float,
    transaction_cost_bps: float,
    rebalance_frequency: int,
    periods_per_year: int,
    max_splits: int | None,
) -> None:
    split_summary = _split_summary(splits)
    params = {
            "model_names": ",".join(model_names),
            "feature_count": _feature_count(features),
            "feature_columns": ",".join(_feature_columns(features)),
            "target_horizon_days": _target_horizon(labels),
            "split_count": predictions["split_id"].nunique(),
            "prediction_rows": len(predictions),
            "max_splits": max_splits,
            "top_k": top_k,
            "top_fraction": top_fraction,
            "transaction_cost_bps": transaction_cost_bps,
            "rebalance_frequency": rebalance_frequency,
            "periods_per_year": periods_per_year,
            "git_commit": _git_commit_hash(),
            **split_summary,
        }
    mlflow.log_params({key: _param_value(value) for key, value in params.items()})


def _log_overall_metrics(
    *,
    mlflow,
    classification_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    portfolio_risk_metrics: pd.DataFrame,
) -> None:
    _log_metric_frame(
        mlflow,
        classification_metrics,
        prefix="classification",
        scope="overall",
    )
    _log_metric_frame(
        mlflow,
        ranking_metrics,
        prefix="ranking",
        scope="overall",
    )
    _log_metric_frame(
        mlflow,
        portfolio_risk_metrics,
        prefix="portfolio",
        scope="overall",
    )


def _log_metric_frame(mlflow, metrics: pd.DataFrame, *, prefix: str, scope: str) -> None:
    if metrics.empty or "scope" not in metrics.columns or "model_name" not in metrics.columns:
        return

    rows = metrics[metrics["scope"] == scope]
    for row in rows.to_dict(orient="records"):
        model_name = _safe_metric_name(str(row["model_name"]))
        for key, value in row.items():
            if key in {"model_name", "scope", "split_id", "date"}:
                continue
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                continue
            mlflow.log_metric(
                f"{prefix}.{model_name}.{_safe_metric_name(key)}",
                float(numeric_value),
            )


def _feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in {"symbol", "date"}]


def _feature_count(features: pd.DataFrame) -> int:
    return len(_feature_columns(features))


def _target_horizon(labels: pd.DataFrame):
    if "label_horizon_days" not in labels.columns:
        return None
    horizons = pd.to_numeric(labels["label_horizon_days"], errors="coerce").dropna().unique()
    return int(horizons[0]) if len(horizons) == 1 else ",".join(map(str, sorted(horizons)))


def _split_summary(splits: pd.DataFrame) -> dict[str, Any]:
    if splits.empty:
        return {}
    ordered = splits.sort_values("split_id")
    return {
        "first_split_id": int(ordered["split_id"].iloc[0]),
        "last_split_id": int(ordered["split_id"].iloc[-1]),
        "first_train_start_date": str(pd.Timestamp(ordered["train_start_date"].iloc[0]).date()),
        "last_validation_end_date": str(
            pd.Timestamp(ordered["validation_end_date"].iloc[-1]).date()
        ),
    }


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _safe_metric_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _param_value(value):
    if value is None:
        return ""
    return value


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment tracking. Install project dependencies "
            "or rerun with --disable-mlflow."
        ) from exc
    return mlflow
