"""Build macro features and merge them into the cumulative feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.macro import (
    MACRO_CHANGE_WINDOWS,
    RATE_CHANGE_WINDOWS,
    build_macro_feature_table,
)

DEFAULT_FEATURE_PATH = Path("data/features/benchmark_relative_features.parquet")
DEFAULT_MACRO_DAILY_PATH = Path("data/processed/macro_daily.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/macro_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build macro features and merge them into market features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to benchmark-relative feature parquet.",
    )
    parser.add_argument(
        "--macro-daily",
        type=Path,
        default=DEFAULT_MACRO_DAILY_PATH,
        help="Path to daily market-aligned macro parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save macro-augmented feature parquet.",
    )
    parser.add_argument(
        "--rate-change-windows",
        type=int,
        nargs="+",
        default=RATE_CHANGE_WINDOWS,
        help="Daily windows for rate and yield-spread changes.",
    )
    parser.add_argument(
        "--macro-change-windows",
        type=int,
        nargs="+",
        default=MACRO_CHANGE_WINDOWS,
        help="Daily windows for CPI, unemployment, and industrial-production changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_macro_feature_table(
        feature_path=args.features,
        macro_daily_path=args.macro_daily,
        output_path=args.output,
        rate_change_windows=args.rate_change_windows,
        macro_change_windows=args.macro_change_windows,
    )

    print(f"Saved macro feature table to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
