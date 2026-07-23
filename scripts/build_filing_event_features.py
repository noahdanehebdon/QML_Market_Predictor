"""Build SEC filing event features and merge them into the cumulative table."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.filing_events import build_filing_event_feature_table

DEFAULT_FEATURE_PATH = Path("data/features/fundamental_features.parquet")
DEFAULT_SUBMISSIONS_PATH = Path("data/processed/sec_submissions.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/filing_event_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SEC filing-date-aware event features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to fundamental feature parquet.",
    )
    parser.add_argument(
        "--submissions",
        type=Path,
        default=DEFAULT_SUBMISSIONS_PATH,
        help="Path to normalized SEC submissions parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save filing event feature parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_filing_event_feature_table(
        feature_path=args.features,
        submissions_path=args.submissions,
        output_path=args.output,
    )

    print(f"Saved filing event feature table to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
