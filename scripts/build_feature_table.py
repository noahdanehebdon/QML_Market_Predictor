"""Build the canonical modeling feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.canonical import build_canonical_feature_table

DEFAULT_FEATURE_PATH = Path("data/features/filing_event_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/feature_table.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and save the canonical modeling feature table."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to the cumulative filing-event feature parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save canonical feature table parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_canonical_feature_table(
        feature_path=args.features,
        output_path=args.output,
    )

    print(f"Saved canonical feature table to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
