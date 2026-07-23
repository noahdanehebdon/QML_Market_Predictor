"""Build date-keyed, leakage-safe market regime labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.features.regimes import build_market_regimes, save_market_regimes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily market regime labels.")
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/feature_table.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/features/market_regimes.parquet")
    )
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--volatility-window", type=int, default=20)
    parser.add_argument("--rate-window", type=int, default=20)
    parser.add_argument("--annualization-factor", type=int, default=252)
    parser.add_argument("--minimum-threshold-history", type=int, default=None)
    parser.add_argument("--curve-flat-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Feature table not found: {args.features}")
    regimes = build_market_regimes(
        pd.read_parquet(args.features),
        benchmark_symbol=args.benchmark,
        volatility_window=args.volatility_window,
        rate_window=args.rate_window,
        annualization_factor=args.annualization_factor,
        minimum_threshold_history=args.minimum_threshold_history,
        curve_flat_tolerance=args.curve_flat_tolerance,
    )
    save_market_regimes(regimes, args.output)
    print(f"Saved market regimes to {args.output}")


if __name__ == "__main__":
    main()
