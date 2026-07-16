"""Analyze aligned QML and classical predictions by market regime."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.regime_analysis import analyze_predictions_by_regime, save_regime_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model performance by market regime.")
    parser.add_argument("--predictions", type=Path, default=Path("reports/qml_comparison/predictions.parquet"))
    parser.add_argument("--regimes", type=Path, default=Path("data/features/market_regimes.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qml_regimes"))
    parser.add_argument("--minimum-rows", type=int, default=50)
    args = parser.parse_args()
    for path in [args.predictions, args.regimes]:
        if not path.exists():
            raise FileNotFoundError(f"Required analysis input not found: {path}")
    result = analyze_predictions_by_regime(
        pd.read_parquet(args.predictions), pd.read_parquet(args.regimes), minimum_rows=args.minimum_rows
    )
    paths = save_regime_analysis(result, args.output_dir)
    print(paths["report"])


if __name__ == "__main__":
    main()
