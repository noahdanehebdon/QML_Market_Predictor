"""Build the unified classical and QML model comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.reporting.model_comparison import (
    build_model_comparison_report,
    save_model_comparison_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified model report.")
    parser.add_argument(
        "--comparison-dir", type=Path, default=Path("reports/qml_comparison")
    )
    parser.add_argument(
        "--regime-metrics",
        type=Path,
        default=Path("reports/qml_regimes/regime_metrics.parquet"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/model_comparison.md"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/model_comparison.csv"),
    )
    parser.add_argument(
        "--regime-output",
        type=Path,
        default=Path("reports/model_comparison_regimes.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "aggregate_metrics": args.comparison_dir / "aggregate_metrics.parquet",
        "ranking_metrics": args.comparison_dir / "ranking_metrics.parquet",
        "portfolio_metrics": args.comparison_dir / "portfolio_metrics.parquet",
        "regime_metrics": args.regime_metrics,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required comparison inputs are missing: " + ", ".join(missing)
        )

    report = build_model_comparison_report(
        **{name: pd.read_parquet(path) for name, path in inputs.items()}
    )
    save_model_comparison_report(
        report,
        markdown_path=args.markdown_output,
        summary_path=args.summary_output,
        regime_path=args.regime_output,
    )
    print(f"Saved model comparison report to {args.markdown_output}")
    print(f"Strongest model: {report.strongest_model or 'not available'}")


if __name__ == "__main__":
    main()
