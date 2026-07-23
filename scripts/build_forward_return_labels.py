"""Build forward return label tables for supervised modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.labels.forward_returns import (
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_LABEL_HORIZON,
    build_forward_return_label_table,
)

DEFAULT_PRICE_PATH = Path("data/processed/prices.parquet")
DEFAULT_OUTPUT_PATH = Path("data/labels/forward_return_labels.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forward excess return labels from processed prices."
    )
    parser.add_argument(
        "--prices",
        type=Path,
        default=DEFAULT_PRICE_PATH,
        help="Path to processed daily prices parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save forward return labels parquet.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_LABEL_HORIZON,
        help="Forward label horizon in trading days.",
    )
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_SYMBOL,
        help="Benchmark symbol for excess return labels.",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="Keep rows without a complete forward return label.",
    )
    parser.add_argument(
        "--neutral-threshold",
        type=float,
        default=0.005,
        help="Absolute excess-return band assigned no neutral-zone class label.",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=20,
        help="Trailing daily observations used to normalize excess returns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = build_forward_return_label_table(
        price_path=args.prices,
        output_path=args.output,
        horizon=args.horizon,
        benchmark_symbol=args.benchmark,
        drop_missing=not args.keep_missing,
        neutral_threshold=args.neutral_threshold,
        volatility_window=args.volatility_window,
    )

    print(f"Saved forward return labels to {args.output}")
    print(f"Rows: {len(labels)}")
    print("\nColumns:")
    print(list(labels.columns))
    print("\nSymbols:")
    print(sorted(labels["symbol"].unique()))
    print("\nTail:")
    print(labels.tail())


if __name__ == "__main__":
    main()
