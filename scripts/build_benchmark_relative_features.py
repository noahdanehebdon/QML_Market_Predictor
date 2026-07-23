"""Build benchmark-relative price features."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from market_qml.features.benchmark import (
    BENCHMARK_WINDOWS,
    EXCESS_RETURN_WINDOWS,
    build_benchmark_relative_features,
)

DEFAULT_FEATURE_PATH = Path("data/features/price_volume_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/features/benchmark_relative_features.parquet")
DEFAULT_UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")


def load_benchmark_symbol(config_path: Path = DEFAULT_UNIVERSE_CONFIG_PATH) -> str:
    if not config_path.exists():
        raise FileNotFoundError(f"Universe config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    universe = config.get("universe")
    if not isinstance(universe, dict) or not universe.get("benchmark"):
        raise ValueError(f"Missing universe.benchmark in {config_path}")

    return str(universe["benchmark"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark-relative price features."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to price volume feature parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save benchmark-relative feature parquet.",
    )
    parser.add_argument(
        "--universe-config",
        type=Path,
        default=DEFAULT_UNIVERSE_CONFIG_PATH,
        help="Path to universe YAML config.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark symbol. Defaults to universe.benchmark.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=BENCHMARK_WINDOWS,
        help="Rolling benchmark windows in trading days.",
    )
    parser.add_argument(
        "--excess-return-windows",
        type=int,
        nargs="+",
        default=EXCESS_RETURN_WINDOWS,
        help="Return windows used for excess-return features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark or load_benchmark_symbol(args.universe_config)
    features = build_benchmark_relative_features(
        feature_path=args.features,
        output_path=args.output,
        benchmark_symbol=benchmark,
        windows=args.windows,
        excess_return_windows=args.excess_return_windows,
    )

    print(f"Saved benchmark-relative features to {args.output}")
    print(f"Benchmark: {benchmark}")
    print(f"Rows: {len(features)}")
    print("\nColumns:")
    print(list(features.columns))
    print("\nSymbols:")
    print(sorted(features["symbol"].unique()))
    print("\nTail:")
    print(features.tail())


if __name__ == "__main__":
    main()
