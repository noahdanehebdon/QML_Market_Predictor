"""Build volume and liquidity features from price feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.volume import (
    VOLUME_WINDOWS,
    build_price_volume_features,
)


DEFAULT_FEATURE_PATH = Path("data/features/price_volatility_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/price_volume_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rolling volume and liquidity features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to price volatility feature parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save volume feature parquet.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=VOLUME_WINDOWS,
        help="Volume windows in trading days.",
    )
    parser.add_argument(
        "--liquidity-min-avg-dollar-volume",
        type=float,
        default=None,
        help="Optional threshold for an average-dollar-volume liquidity flag.",
    )
    parser.add_argument(
        "--liquidity-window",
        type=int,
        default=20,
        help="Window used for the optional liquidity flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_price_volume_features(
        feature_path=args.features,
        output_path=args.output,
        windows=args.windows,
        liquidity_min_avg_dollar_volume=args.liquidity_min_avg_dollar_volume,
        liquidity_window=args.liquidity_window,
    )

    print(f"Saved price volume features to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
