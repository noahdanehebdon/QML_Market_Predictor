"""Run baseline models across all walk-forward splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from market_qml.reporting.baseline_evidence import compare_to_naive_baselines

from market_qml.backtest.classification_metrics import (
    CLASSIFICATION_METRIC_COLUMNS,
    evaluate_classification_metrics,
    save_classification_metrics,
)
from market_qml.backtest.portfolio import (
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_RETURN_HORIZON_DAYS,
    DEFAULT_TRANSACTION_COST_BPS,
    TRADING_DAYS_PER_YEAR,
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
from market_qml.models.artifacts import (
    resolve_git_sha,
    save_artifact_manifest,
    save_model_artifact,
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
from market_qml.models.naive_rankers import (
    TARGET as NAIVE_RANK_TARGET,
    train_linear_rank,
    train_momentum_rank,
    train_random_rank,
    train_sector_neutral_rank,
    train_sign_rank,
)
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
from market_qml.models.tuned_gradient_boosting import (
    DEFAULT_TARGET_COLUMN as TUNED_GRADIENT_BOOSTING_TARGET_COLUMN,
    MODEL_NAME as TUNED_GRADIENT_BOOSTING_MODEL_NAME,
    train_tuned_gradient_boosting_regressor,
)
from market_qml.models.xgboost_baselines import (
    CLASSIFIER_NAME as XGBOOST_CLASSIFIER_NAME,
    RANKER_NAME as XGBOOST_RANKER_NAME,
    RANK_TARGET as XGBOOST_RANK_TARGET,
    train_xgboost_classifier,
    train_xgboost_ranker,
)
from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.pca import fit_pca
from market_qml.qml.vqc import MODEL_NAME as VQC_MODEL_NAME
from market_qml.qml.vqc import train_vqc
from market_qml.utils.mlflow_tracking import DEFAULT_EXPERIMENT_NAME
from market_qml.utils.mlflow_tracking import log_walk_forward_backtest_run


DEFAULT_OUTPUT_DIR = Path("reports/backtests")
DEFAULT_QML_N_COMPONENTS = 8
VOL_NORMALIZED_GRADIENT_BOOSTING_MODEL_NAME = (
    "vol_normalized_gradient_boosting_regressor"
)
VOL_NORMALIZED_TARGET_COLUMN = "vol_normalized_excess_return_5d"


@dataclass(frozen=True)
class ModelSpec:
    """Training configuration for one walk-forward baseline."""

    target_column: str
    train: Callable
    model_family: str = "classical"


@dataclass(frozen=True)
class WalkForwardPredictionResult:
    """Walk-forward predictions plus optional model-specific diagnostics."""

    predictions: pd.DataFrame
    training_loss: pd.DataFrame
    validation_metrics: pd.DataFrame
    selection_diagnostics: pd.DataFrame
    artifact_records: list[dict]


MODEL_REGISTRY = {
    "sign_rank": ModelSpec(target_column=NAIVE_RANK_TARGET, train=train_sign_rank),
    "momentum_rank": ModelSpec(target_column=NAIVE_RANK_TARGET, train=train_momentum_rank),
    "random_rank": ModelSpec(target_column=NAIVE_RANK_TARGET, train=train_random_rank),
    "sector_neutral_rank": ModelSpec(target_column=NAIVE_RANK_TARGET, train=train_sector_neutral_rank),
    "linear_rank": ModelSpec(target_column=NAIVE_RANK_TARGET, train=train_linear_rank),
    XGBOOST_CLASSIFIER_NAME: ModelSpec(
        target_column=DEFAULT_TARGET_COLUMN,
        train=train_xgboost_classifier,
    ),
    XGBOOST_RANKER_NAME: ModelSpec(
        target_column=XGBOOST_RANK_TARGET,
        train=train_xgboost_ranker,
    ),
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
    TUNED_GRADIENT_BOOSTING_MODEL_NAME: ModelSpec(
        target_column=TUNED_GRADIENT_BOOSTING_TARGET_COLUMN,
        train=train_tuned_gradient_boosting_regressor,
    ),
    VOL_NORMALIZED_GRADIENT_BOOSTING_MODEL_NAME: ModelSpec(
        target_column=VOL_NORMALIZED_TARGET_COLUMN,
        train=lambda data, split_id: train_gradient_boosting_regressor(
            data,
            model_name=VOL_NORMALIZED_GRADIENT_BOOSTING_MODEL_NAME,
            split_id=split_id,
            l2_regularization=0.1,
        ),
    ),
    VQC_MODEL_NAME: ModelSpec(
        target_column=DEFAULT_TARGET_COLUMN,
        train=train_vqc,
        model_family="qml",
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
        "--universe-membership",
        type=Path,
        default=None,
        help="Optional point-in-time membership parquet used to filter model rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save backtest outputs.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for fitted model bundles (defaults inside output-dir).",
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
        default=DEFAULT_TRANSACTION_COST_BPS,
        help="One-way transaction cost in basis points applied to turnover.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=DEFAULT_REBALANCE_FREQUENCY,
        help="Number of prediction dates between portfolio rebalances.",
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
        universe_membership=(
            pd.read_parquet(args.universe_membership)
            if args.universe_membership is not None
            else None
        ),
        model_names=args.models,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        max_splits=args.max_splits,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_frequency=args.rebalance_frequency,
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
    universe_membership: pd.DataFrame | None = None,
    model_names: list[str],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    artifact_dir: str | Path | None = None,
    max_splits: int | None = None,
    top_k: int | None = None,
    top_fraction: float = 0.1,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    rebalance_frequency: int = DEFAULT_REBALANCE_FREQUENCY,
    return_horizon_days: int = DEFAULT_RETURN_HORIZON_DAYS,
    enable_mlflow: bool = False,
    mlflow_experiment: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_run_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
) -> dict[str, Path]:
    """Train selected models over walk-forward splits and save report outputs."""
    if max_splits is not None and max_splits <= 0:
        raise ValueError("max_splits must be positive when provided.")

    output_dir = Path(output_dir)
    artifact_dir = (
        Path(artifact_dir) if artifact_dir else output_dir / "model_artifacts"
    )
    selected_splits = _selected_splits(splits, max_splits=max_splits)
    run_config = {
        "models": model_names,
        "max_splits": max_splits,
        "top_k": top_k,
        "top_fraction": top_fraction,
        "transaction_cost_bps": transaction_cost_bps,
        "rebalance_frequency": rebalance_frequency,
        "return_horizon_days": return_horizon_days,
    }
    prediction_result = _walk_forward_predictions(
        features=features,
        labels=labels,
        universe_membership=universe_membership,
        splits=selected_splits,
        model_names=model_names,
        artifact_dir=artifact_dir,
        run_config=run_config,
        git_sha=resolve_git_sha(),
    )
    predictions = prediction_result.predictions
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
        return_horizon_days=return_horizon_days,
    )
    portfolio_risk_metrics = summarize_portfolio_risk(
        portfolio_returns,
    )
    baseline_evidence = compare_to_naive_baselines(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "predictions": output_dir / "predictions.parquet",
        "classification_metrics": output_dir / "classification_metrics.parquet",
        "ranking_metrics": output_dir / "ranking_metrics.parquet",
        "portfolio_backtest": output_dir / "portfolio_backtest.parquet",
        "portfolio_risk_metrics": output_dir / "portfolio_risk_metrics.parquet",
        "baseline_evidence": output_dir / "baseline_evidence.parquet",
        "artifact_manifest": save_artifact_manifest(
            artifact_dir, prediction_result.artifact_records
        ),
    }
    if not prediction_result.training_loss.empty:
        output_paths["training_loss"] = output_dir / "training_loss.parquet"
    if not prediction_result.validation_metrics.empty:
        output_paths["validation_metrics"] = output_dir / "validation_metrics.parquet"
    if not prediction_result.selection_diagnostics.empty:
        output_paths["selection_diagnostics"] = (
            output_dir / "selection_diagnostics.parquet"
        )

    predictions.to_parquet(output_paths["predictions"], index=False)
    baseline_evidence.to_parquet(output_paths["baseline_evidence"], index=False)
    if "training_loss" in output_paths:
        prediction_result.training_loss.to_parquet(
            output_paths["training_loss"],
            index=False,
        )
    if "validation_metrics" in output_paths:
        prediction_result.validation_metrics.to_parquet(
            output_paths["validation_metrics"],
            index=False,
        )
    if "selection_diagnostics" in output_paths:
        prediction_result.selection_diagnostics.to_parquet(
            output_paths["selection_diagnostics"], index=False
        )
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
            periods_per_year=TRADING_DAYS_PER_YEAR / rebalance_frequency,
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
    universe_membership: pd.DataFrame | None,
    splits: pd.DataFrame,
    model_names: list[str],
    artifact_dir: Path,
    run_config: dict,
    git_sha: str,
) -> WalkForwardPredictionResult:
    frames = []
    training_loss_frames = []
    validation_metric_frames = []
    selection_diagnostic_frames = []
    artifact_records = []
    for model_name in model_names:
        spec = MODEL_REGISTRY[model_name]
        for split in splits.itertuples(index=False):
            datasets = build_train_validation_datasets(
                features=features,
                labels=labels,
                universe_membership=universe_membership,
                target_column=spec.target_column,
                train_start_date=split.train_start_date,
                train_end_date=split.train_end_date,
                validation_start_date=split.validation_start_date,
                validation_end_date=split.validation_end_date,
            )
            preprocessed = fit_transform_train_validation(datasets)
            pca = None
            if spec.model_family == "qml":
                qml_sample, pca = _build_qml_split_sample(
                    preprocessed=preprocessed,
                    split_id=int(split.split_id),
                )
                result = spec.train(
                    qml_sample,
                    split_id=int(split.split_id),
                )
                training_loss_frames.append(result.training_loss)
                validation_metric_frames.append(result.validation_metrics)
            else:
                result = spec.train(
                    preprocessed,
                    split_id=int(split.split_id),
                )
                if hasattr(result, "selection_diagnostics"):
                    selection_diagnostic_frames.append(result.selection_diagnostics)
            record = save_model_artifact(
                root=artifact_dir,
                model_name=model_name,
                split_id=int(split.split_id),
                model=result.model,
                preprocessor=preprocessed.preprocessor,
                pca=pca,
                result=result,
                train_metadata=preprocessed.train.metadata,
                validation_metadata=preprocessed.validation.metadata,
                target_column=spec.target_column,
                run_config=run_config,
                git_sha=git_sha,
            )
            artifact_records.append(record)
            model_predictions = result.predictions.copy()
            model_predictions["artifact_id"] = record["artifact_id"]
            frames.append(model_predictions)

    if not frames:
        raise ValueError("No prediction rows were produced.")

    return WalkForwardPredictionResult(
        predictions=pd.concat(frames, ignore_index=True),
        training_loss=(
            pd.concat(training_loss_frames, ignore_index=True)
            if training_loss_frames
            else pd.DataFrame()
        ),
        validation_metrics=(
            pd.concat(validation_metric_frames, ignore_index=True)
            if validation_metric_frames
            else pd.DataFrame()
        ),
        selection_diagnostics=(
            pd.concat(selection_diagnostic_frames, ignore_index=True)
            if selection_diagnostic_frames
            else pd.DataFrame()
        ),
        artifact_records=artifact_records,
    )


def _build_qml_split_sample(
    *,
    preprocessed,
    split_id: int,
):
    pca = fit_pca(preprocessed.train.X, n_components=DEFAULT_QML_N_COMPONENTS)
    train_rows = _pca_rows(
        X=preprocessed.train.X,
        y=preprocessed.train.y,
        metadata=preprocessed.train.metadata,
        pca=pca,
        split_id=split_id,
        sample_role="train",
    )
    validation_rows = _pca_rows(
        X=preprocessed.validation.X,
        y=preprocessed.validation.y,
        metadata=preprocessed.validation.metadata,
        pca=pca,
        split_id=split_id,
        sample_role="validation",
    )
    qml_sample = pd.concat([train_rows, validation_rows], ignore_index=True)
    return build_qml_train_validation(qml_sample, split_id=split_id), pca


def _pca_rows(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    pca,
    split_id: int,
    sample_role: str,
) -> pd.DataFrame:
    component_columns = [
        f"pca_{component_index:02d}"
        for component_index in range(DEFAULT_QML_N_COMPONENTS)
    ]
    component_frame = pd.DataFrame(
        pca.transform(X),
        columns=component_columns,
        index=X.index,
    )
    result = metadata.copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["split_id"] = split_id
    result["sample_role"] = sample_role
    result["target"] = pd.to_numeric(y, errors="coerce").to_numpy()
    result = pd.concat(
        [result, component_frame.reset_index(drop=True)],
        axis=1,
    )
    if result["date"].isna().any():
        raise ValueError("QML walk-forward PCA rows contain invalid dates.")
    if result["target"].isna().any():
        raise ValueError("QML walk-forward PCA rows contain invalid targets.")
    return result


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
