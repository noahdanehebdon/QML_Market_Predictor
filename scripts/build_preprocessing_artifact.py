"""Fit and save train-only preprocessing artifacts."""

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


DEFAULT_OUTPUT_DIR = Path("artifacts/preprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit preprocessing on a walk-forward training split."
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
        help="Walk-forward split id to fit preprocessing for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for fitted preprocessing artifacts.",
    )
    parser.add_argument(
        "--max-missing-feature-fraction",
        type=float,
        default=None,
        help="Optional row filter before preprocessing.",
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
        max_missing_feature_fraction=args.max_missing_feature_fraction,
    )
    preprocessed = fit_transform_train_validation(datasets)

    output_path = args.output_dir / f"preprocessor_split_{args.split_id:03d}.pkl"
    save_preprocessor(preprocessed.preprocessor, output_path)

    print(f"Saved preprocessing artifact to {output_path}")
    print(f"Split: {args.split_id}")
    print(f"Train X shape: {preprocessed.train.X.shape}")
    print(f"Validation X shape: {preprocessed.validation.X.shape}")
    print(f"Feature columns: {len(preprocessed.preprocessor.feature_columns)}")


if __name__ == "__main__":
    main()
