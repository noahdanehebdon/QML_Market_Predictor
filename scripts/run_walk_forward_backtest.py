"""Run baseline models across all walk-forward splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from market_qml.backtest.classification_metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    evaluate_classification_metrics,
    save_classification_metrics,
)
from market_qml.backtest.portfolio import (
    run_portfolio_backtest,
    save_portfolio_returns,
    save_portfolio_risk_metrics,
    summarize_portfolio_risk,
)
from market_qml.backtest.ranking_metrics import (
    evaluate_ranking_metrics,
    save_ranking_metrics,
)
from market_qml.backtest.splits import DEFAULT_SPLIT_OUTPUT_PATH
from market_qml.models.dataset import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_TARGET_COLUMN,
    build_train_validation_datasets,
)
from market_qml.models.elastic_net import (
    DEFAULT_TARGET_COLUMN as ELASTIC_NET_TARGET_COLUMN,
    MODEL_NAME as ELASTIC_NET_MODEL_NAME,
    train_elastic_net,
)
from market_qml.models.gradient_boosting import (
    MODEL_NAME as GRADIENT_BOOSTING_MODEL_NAME,
    train_gradient_boosting,
)
from market_qml.models.gradient_boosting_regressor import (
    DEFAULT_TARGET_COLUMN as GRADIENT_BOOSTING_REGRESSOR_TARGET_COLUMN,
    MODEL_NAME as GRADIENT_BOOSTING_REGRESSOR_MODEL_NAME,
    train_gradient_boosting_regressor,
)
from market_qml.models.huber_regression import (
    DEFAULT_TARGET_COLUMN as HUBER_TARGET_COLUMN,
    MODEL_NAME as HUBER_REGRESSION_MODEL_NAME,
    train_huber_regression,
)
from market_qml.models.logistic_regression import (
    MODEL_NAME as LOGISTIC_REGRESSION_MODEL_NAME,
    train_logistic_regression,
)
from market_qml.models.preprocessing import fit_transform_train_validation
from market_qml.models.random_forest import (
    MODEL_NAME as RANDOM_FOREST_MODEL_NAME,
    train_random_forest,
)
from market_qml.models.random_forest_regressor import (
    DEFAULT_TARGET_COLUMN as RANDOM_FOREST_REGRESSOR_TARGET_COLUMN,
    MODEL_NAME as RANDOM_FOREST_REGRESSOR_MODEL_NAME,
    train_random_forest_regressor,
)
from market_qml.models.ridge_regression import (
    DEFAULT_TARGET_COLUMN as RIDGE_TARGET_COLUMN,
    MODEL_NAME as RIDGE_REGRESSION_MODEL_NAME,
    train_ridge_regression,
)
from market_qml.utils.mlflow_tracking import DEFAULT_EXPERIMENT_NAME
from market_qml.utils.mlflow_tracking import log_walk_forward_backtest_run


DEFAULT_OUTPUT_DIR = Path("reports/backtests")


@dataclass(frozen=True)
class ModelSpec:
    """Training configuration for one walk-forward baseline."""

    target_column: str
    train: Callable


MODEL_REGISTRY = {
    LOGISTIC_REGRESSION_MODEL_NAME: ModelSpec(
        target_column=DEFAULT_TARGET_COLUMN,
        train=train_logistic_regression,
    ),
    ELASTIC_NET_MODEL_NAME: ModelSpec(
        target_column=ELASTIC_NET_TARGET_COLUMN,
        train=train_elastic_net,
    ),
    RIDGE_REGRESSION_MODEL_NAME: ModelSpec(
        target_column=RIDGE_TARGET_COLUMN,
        train=train_ridge_regression,
    ),
    HUBER_REGRESSION_MODEL_NAME: ModelSpec(
        target_column=HUBER_TARGET_COLUMN,
        train=train_huber_regression,
    ),
    RANDOM_FOREST_MODEL_NAME: ModelSpec(
        target_column=DEFAULT_TARGET_COLUMN,
        train=train_random_forest,
    ),
    RANDOM_FOREST_REGRESSOR_MODEL_NAME: ModelSpec(
        target_column=RANDOM_FOREST_REGRESSOR_TARGET_COLUMN,
        train=train_random_forest_regressor,
    ),
    GRADIENT_BOOSTING_MODEL_NAME: ModelSpec(
        target_column=DEFAULT_TARGET_COLUMN,
        train=train_gradient_boosting,
    ),
    GRADIENT_BOOSTING_REGRESSOR_MODEL_NAME: ModelSpec(
        target_column=GRADIENT_BOOSTING_REGRESSOR_TARGET_COLUMN,
        train=train_gradient_boosting_regressor,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline models across all walk-forward splits."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to canonical feature table parquet.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help="Path to forward return label table parquet.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT_PATH,
        help="Path to walk-forward split metadata parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save backtest outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=sorted(MODEL_REGISTRY),
        help="Model names to run.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Optional cap on number of splits for quick smoke runs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top-ranked names to select each date for portfolio backtest.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Top fraction for ranking and portfolio metrics.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in basis points applied to turnover.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=5,
        help="Number of prediction dates between portfolio rebalances.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualization periods used for volatility and Sharpe ratio.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name for tracking backtest runs.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Optional MLflow tracking URI.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Skip MLflow logging for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_walk_forward_backtest(
        features=pd.read_parquet(args.features),
        labels=pd.read_parquet(args.labels),
        splits=pd.read_parquet(args.splits),
        model_names=args.models,
        output_dir=args.output_dir,
        max_splits=args.max_splits,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_frequency=args.rebalance_frequency,
        periods_per_year=args.periods_per_year,
        enable_mlflow=not args.disable_mlflow,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )

    print(f"Saved walk-forward backtest outputs to {args.output_dir}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_walk_forward_backtest(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    model_names: list[str],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_splits: int | None = None,
    top_k: int | None = None,
    top_fraction: float = 0.1,
    transaction_cost_bps: float = 0.0,
    rebalance_frequency: int = 5,
    periods_per_year: int = 252,
    enable_mlflow: bool = False,
    mlflow_experiment: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_run_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Path]:
    """Train selected models over walk-forward splits and save report outputs."""
    if max_splits is not None and max_splits <= 0:
        raise ValueError("max_splits must be positive when provided.")

    selected_splits = _selected_splits(splits, max_splits=max_splits)
    predictions = _walk_forward_predictions(
        features=features,
        labels=labels,
        splits=selected_splits,
        model_names=model_names,
    )
    binary_predictions = _binary_predictions(predictions)

    classification_metrics = (
        evaluate_classification_metrics(binary_predictions)
        if not binary_predictions.empty
        else pd.DataFrame(columns=CLASSIFICATION_METRIC_COLUMNS)
    )
    ranking_metrics = evaluate_ranking_metrics(
        predictions,
        top_fraction=min(top_fraction, 0.5),
    )
    portfolio_returns = run_portfolio_backtest(
        predictions,
        top_k=top_k,
        top_fraction=top_fraction,
        transaction_cost_bps=transaction_cost_bps,
        rebalance_frequency=rebalance_frequency,
    )
    portfolio_risk_metrics = summarize_portfolio_risk(
        portfolio_returns,
        periods_per_year=periods_per_year,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "predictions": output_dir / "predictions.parquet",
        "classification_metrics": output_dir / "classification_metrics.parquet",
        "ranking_metrics": output_dir / "ranking_metrics.parquet",
        "portfolio_backtest": output_dir / "portfolio_backtest.parquet",
        "portfolio_risk_metrics": output_dir / "portfolio_risk_metrics.parquet",
    }
    predictions.to_parquet(output_paths["predictions"], index=False)
    save_classification_metrics(
        classification_metrics,
        output_paths["classification_metrics"],
    )
    save_ranking_metrics(ranking_metrics, output_paths["ranking_metrics"])
    save_portfolio_returns(portfolio_returns, output_paths["portfolio_backtest"])
    save_portfolio_risk_metrics(
        portfolio_risk_metrics,
        output_paths["portfolio_risk_metrics"],
    )
    if enable_mlflow:
        log_walk_forward_backtest_run(
            output_paths=output_paths,
            predictions=predictions,
            splits=selected_splits,
            features=features,
            labels=labels,
            classification_metrics=classification_metrics,
            ranking_metrics=ranking_metrics,
            portfolio_risk_metrics=portfolio_risk_metrics,
            model_names=model_names,
            top_k=top_k,
            top_fraction=top_fraction,
            transaction_cost_bps=transaction_cost_bps,
            rebalance_frequency=rebalance_frequency,
            periods_per_year=periods_per_year,
            max_splits=max_splits,
            experiment_name=mlflow_experiment,
            run_name=mlflow_run_name,
            tracking_uri=mlflow_tracking_uri,
        )

    return output_paths


def _walk_forward_predictions(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    frames = []
    for model_name in model_names:
        spec = MODEL_REGISTRY[model_name]
        for split in splits.itertuples(index=False):
            datasets = build_train_validation_datasets(
                features=features,
                labels=labels,
                target_column=spec.target_column,
                train_start_date=split.train_start_date,
                train_end_date=split.train_end_date,
                validation_start_date=split.validation_start_date,
                validation_end_date=split.validation_end_date,
            )
            preprocessed = fit_transform_train_validation(datasets)
            result = spec.train(
                preprocessed,
                split_id=int(split.split_id),
            )
            frames.append(result.predictions)

    if not frames:
        raise ValueError("No prediction rows were produced.")

    return pd.concat(frames, ignore_index=True)


def _selected_splits(splits: pd.DataFrame, *, max_splits: int | None) -> pd.DataFrame:
    if splits.empty:
        raise ValueError("Walk-forward split table is empty.")

    selected = splits.sort_values("split_id").reset_index(drop=True)
    if max_splits is not None:
        selected = selected.head(max_splits)
    return selected


def _binary_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in predictions.groupby("model_name", sort=True):
        values = pd.to_numeric(group["y_true"], errors="coerce").dropna().unique()
        if set(values).issubset({0, 1}):
            frames.append(group)
    if not frames:
        return pd.DataFrame(columns=predictions.columns)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    main()
