"""Build SEC fundamental features and merge them into the cumulative table."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.fundamentals import build_fundamental_feature_table


DEFAULT_FEATURE_PATH = Path("data/features/macro_features.parquet")
DEFAULT_FUNDAMENTALS_PATH = Path("data/processed/fundamentals.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/fundamental_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SEC filing-date-aware fundamental features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to macro-augmented feature parquet.",
    )
    parser.add_argument(
        "--fundamentals",
        type=Path,
        default=DEFAULT_FUNDAMENTALS_PATH,
        help="Path to normalized SEC companyfacts parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save fundamental feature parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_fundamental_feature_table(
        feature_path=args.features,
        fundamentals_path=args.fundamentals,
        output_path=args.output,
    )

    print(f"Saved fundamental feature table to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
