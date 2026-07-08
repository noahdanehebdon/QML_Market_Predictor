"""Train gradient boosting classifier baseline on one walk-forward split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.backtest.splits import DEFAULT_SPLIT_OUTPUT_PATH
from market_qml.models.dataset import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    build_train_validation_datasets,
)
from market_qml.models.gradient_boosting import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARAMETERS_PATH,
    DEFAULT_PREDICTION_PATH,
    save_gradient_boosting_model,
    save_metrics,
    save_model_parameters,
    save_predictions,
    train_gradient_boosting,
)
from market_qml.models.preprocessing import (
    fit_transform_train_validation,
    save_preprocessor,
)


DEFAULT_PREPROCESSOR_PATH = Path("artifacts/preprocessing/gradient_boosting_split_000.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train gradient boosting classifier baseline on a walk-forward split."
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
        "--split-id",
        type=int,
        default=0,
        help="Walk-forward split id to train on.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="Path to save validation prediction parquet.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to save validation metrics parquet.",
    )
    parser.add_argument(
        "--parameters-output",
        type=Path,
        default=DEFAULT_PARAMETERS_PATH,
        help="Path to save model parameter JSON.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save fitted model artifact.",
    )
    parser.add_argument(
        "--preprocessor-output",
        type=Path,
        default=DEFAULT_PREPROCESSOR_PATH,
        help="Path to save fitted preprocessing artifact.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient boosting learning rate.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum boosting iterations.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=31,
        help="Maximum leaf nodes per tree.",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=0.0,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples required at each leaf.",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=255,
        help="Maximum bins used by histogram boosting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.splits.exists():
        raise FileNotFoundError(
            f"Walk-forward splits not found: {args.splits}. "
            "Run python -m scripts.build_walk_forward_splits first."
        )

    splits = pd.read_parquet(args.splits)
    split_rows = splits[splits["split_id"] == args.split_id]
    if split_rows.empty:
        raise KeyError(f"Split id not found: {args.split_id}")

    split = split_rows.iloc[0]
    datasets = build_train_validation_datasets(
        features=pd.read_parquet(args.features),
        labels=pd.read_parquet(args.labels),
        train_start_date=split["train_start_date"],
        train_end_date=split["train_end_date"],
        validation_start_date=split["validation_start_date"],
        validation_end_date=split["validation_end_date"],
    )
    preprocessed = fit_transform_train_validation(datasets)
    result = train_gradient_boosting(
        preprocessed,
        split_id=args.split_id,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        min_samples_leaf=args.min_samples_leaf,
        max_bins=args.max_bins,
    )

    save_preprocessor(preprocessed.preprocessor, args.preprocessor_output)
    save_gradient_boosting_model(result.model, args.model_output)
    save_predictions(result.predictions, args.predictions_output)
    save_metrics(result.metrics, args.metrics_output)
    save_model_parameters(result.parameters, args.parameters_output)

    print(f"Saved predictions to {args.predictions_output}")
    print(f"Saved metrics to {args.metrics_output}")
    print(f"Saved model parameters to {args.parameters_output}")
    print(f"Saved model to {args.model_output}")
    print(f"Saved preprocessor to {args.preprocessor_output}")
    print(f"Rows: {len(result.predictions)}")
    print("\nPrediction columns:")
    print(list(result.predictions.columns))
    print("\nMetrics:")
    print(result.metrics)
    print("\nPrediction tail:")
    print(result.predictions.tail())


if __name__ == "__main__":
    main()
