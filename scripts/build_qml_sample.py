"""Build a reproducible sampled QML dataset from compressed QML features."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.qml.pca import DEFAULT_QML_PCA_FEATURE_PATH
from market_qml.qml.sampling import (
    DEFAULT_MAX_TRAIN_ROWS_PER_SPLIT,
    DEFAULT_MAX_VALIDATION_ROWS_PER_SPLIT,
    DEFAULT_QML_SAMPLE_METADATA_PATH,
    DEFAULT_QML_SAMPLE_PATH,
    DEFAULT_RANDOM_SEED,
    build_qml_sample,
    load_qml_features,
    save_qml_sample,
    save_qml_sample_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible sampled QML dataset."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_QML_PCA_FEATURE_PATH,
        help="Path to PCA-compressed QML feature rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QML_SAMPLE_PATH,
        help="Path to save sampled QML rows.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_QML_SAMPLE_METADATA_PATH,
        help="Path to save sampling metadata.",
    )
    parser.add_argument(
        "--max-train-rows-per-split",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS_PER_SPLIT,
        help="Maximum train rows per split.",
    )
    parser.add_argument(
        "--max-validation-rows-per-split",
        type=int,
        default=DEFAULT_MAX_VALIDATION_ROWS_PER_SPLIT,
        help="Maximum validation rows per split.",
    )
    parser.add_argument(
        "--max-dates-per-split-role",
        type=int,
        default=None,
        help="Optional maximum dates per split/sample role.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Optional maximum number of symbols sampled globally.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--no-balance-classes",
        action="store_true",
        help="Disable class balancing for binary targets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_qml_sample(
        load_qml_features(args.features),
        max_train_rows_per_split=args.max_train_rows_per_split,
        max_validation_rows_per_split=args.max_validation_rows_per_split,
        max_dates_per_split_role=args.max_dates_per_split_role,
        max_symbols=args.max_symbols,
        balance_classes=not args.no_balance_classes,
        random_seed=args.random_seed,
    )
    save_qml_sample(result.sample, args.output)
    save_qml_sample_metadata(result.metadata, args.metadata_output)

    print(f"Saved sampled QML dataset to {args.output}")
    print(f"Saved sampling metadata to {args.metadata_output}")
    print(f"Rows: {len(result.sample)}")
    print(f"Splits: {sorted(result.sample['split_id'].unique().tolist())}")
    print(f"Roles: {sorted(result.sample['sample_role'].unique().tolist())}")
    print("\nMetadata:")
    print(result.metadata)


if __name__ == "__main__":
    main()
