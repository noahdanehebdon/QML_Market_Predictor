"""Generate the latest cross-sectional equity signal report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.models.logistic_regression import DEFAULT_MODEL_PATH
from market_qml.models.preprocessing import load_preprocessor
from market_qml.reporting.daily_signal import (
    build_daily_signal_report,
    load_model,
    save_daily_signal_report,
)


DEFAULT_FEATURE_PATH = Path("data/features/feature_table.parquet")
DEFAULT_PREPROCESSOR_PATH = Path(
    "artifacts/preprocessing/logistic_regression_split_000.pkl"
)
DEFAULT_MARKDOWN_PATH = Path("reports/daily_signal.md")
DEFAULT_CSV_PATH = Path("reports/daily_signal.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the daily signal report.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--preprocessor", type=Path, default=DEFAULT_PREPROCESSOR_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--model-name", default="logistic_regression")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--bottom-n", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_daily_signal_report(
        pd.read_parquet(args.features),
        model=load_model(args.model),
        preprocessor=load_preprocessor(args.preprocessor),
        model_name=args.model_name,
        benchmark=args.benchmark,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
    )
    save_daily_signal_report(
        report,
        markdown_path=args.markdown_output,
        csv_path=args.csv_output,
    )
    print(f"Saved daily signal Markdown to {args.markdown_output}")
    print(f"Saved daily signal CSV to {args.csv_output}")
    print(f"Signal date: {report.signal_date.date().isoformat()}")
    print(f"Symbols ranked: {len(report.signals)}")


if __name__ == "__main__":
    main()
