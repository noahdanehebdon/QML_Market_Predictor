"""Train random forest classifier baseline on one walk-forward split."""

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
from market_qml.models.preprocessing import (
    fit_transform_train_validation,
    save_preprocessor,
)
from market_qml.models.random_forest import (
    DEFAULT_IMPORTANCE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_PATH,
    save_feature_importance,
    save_predictions,
    save_random_forest_model,
    train_random_forest,
)


DEFAULT_PREPROCESSOR_PATH = Path("artifacts/preprocessing/random_forest_split_000.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train random forest classifier baseline on a walk-forward split."
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
        "--feature-importance-output",
        type=Path,
        default=DEFAULT_IMPORTANCE_PATH,
        help="Path to save feature importance parquet.",
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
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth. Use 0 for unlimited depth.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=10,
        help="Minimum samples required at each leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Number of features considered at each split.",
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
    result = train_random_forest(
        preprocessed,
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    )

    save_preprocessor(preprocessed.preprocessor, args.preprocessor_output)
    save_random_forest_model(result.model, args.model_output)
    save_predictions(result.predictions, args.predictions_output)
    save_feature_importance(result.feature_importance, args.feature_importance_output)

    print(f"Saved predictions to {args.predictions_output}")
    print(f"Saved feature importance to {args.feature_importance_output}")
    print(f"Saved model to {args.model_output}")
    print(f"Saved preprocessor to {args.preprocessor_output}")
    print(f"Rows: {len(result.predictions)}")
    print("\nPrediction columns:")
    print(list(result.predictions.columns))
    print("\nTop feature importances:")
    print(result.feature_importance.head(10))
    print("\nPrediction tail:")
    print(result.predictions.tail())


if __name__ == "__main__":
    main()
