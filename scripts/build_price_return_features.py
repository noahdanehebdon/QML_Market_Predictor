"""Build price return features from processed daily prices."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.features.returns import (
    RETURN_WINDOWS,
    build_price_return_features,
)

DEFAULT_PRICE_PATH = Path("data/processed/prices.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/price_return_features.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build backward-looking price return features."
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
        help="Path to save price return feature parquet.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=RETURN_WINDOWS,
        help="Return windows in trading days.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_price_return_features(
        price_path=args.prices,
        output_path=args.output,
        windows=args.windows,
    )

    print(f"Saved price return features to {args.output}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
