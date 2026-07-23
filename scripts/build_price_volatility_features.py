"""Build rolling realized volatility features from price return features."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.volatility import (
    TRADING_DAYS_PER_YEAR,
    VOLATILITY_WINDOWS,
    build_price_volatility_features,
)

DEFAULT_FEATURE_PATH = Path("data/features/price_return_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/price_volatility_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annualized rolling realized volatility features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to price return feature parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save volatility feature parquet.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=VOLATILITY_WINDOWS,
        help="Volatility windows in trading days.",
    )
    parser.add_argument(
        "--annualization-factor",
        type=int,
        default=TRADING_DAYS_PER_YEAR,
        help="Trading periods per year used to annualize volatility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_price_volatility_features(
        feature_path=args.features,
        output_path=args.output,
        windows=args.windows,
        annualization_factor=args.annualization_factor,
    )

    print(f"Saved price volatility features to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
