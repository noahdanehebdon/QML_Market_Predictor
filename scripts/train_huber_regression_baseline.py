"""Train Huber regression baseline on one walk-forward split."""

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
from market_qml.models.huber_regression import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_PATH,
    DEFAULT_TARGET_COLUMN,
    save_huber_regression_model,
    save_predictions,
    train_huber_regression,
)
from market_qml.models.preprocessing import (
    fit_transform_train_validation,
    save_preprocessor,
)

DEFAULT_PREPROCESSOR_PATH = Path(
    "artifacts/preprocessing/huber_regression_split_000.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Huber regression baseline on a walk-forward split."
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
        "--target",
        default=DEFAULT_TARGET_COLUMN,
        help="Continuous target column from the label table.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.35,
        help="Huber loss threshold where squared loss transitions to absolute loss.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0001,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
        help="Path to save validation prediction parquet.",
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
        target_column=args.target,
        train_start_date=split["train_start_date"],
        train_end_date=split["train_end_date"],
        validation_start_date=split["validation_start_date"],
        validation_end_date=split["validation_end_date"],
    )
    preprocessed = fit_transform_train_validation(datasets)
    result = train_huber_regression(
        preprocessed,
        split_id=args.split_id,
        epsilon=args.epsilon,
        alpha=args.alpha,
        max_iter=args.max_iter,
    )

    save_preprocessor(preprocessed.preprocessor, args.preprocessor_output)
    save_huber_regression_model(result.model, args.model_output)
    save_predictions(result.predictions, args.predictions_output)

    print(f"Saved predictions to {args.predictions_output}")
    print(f"Saved model to {args.model_output}")
    print(f"Saved preprocessor to {args.preprocessor_output}")
    print(f"Rows: {len(result.predictions)}")
    print("\nColumns:")
    print(list(result.predictions.columns))
    print("\nHead:")
    print(result.predictions.head())
    print("\nTail:")
    print(result.predictions.tail())


if __name__ == "__main__":
    main()
