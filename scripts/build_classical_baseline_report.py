"""Build a classical baseline comparison report from backtest outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.reporting.classical_baselines import (
    build_classical_baseline_comparison,
    render_classical_baseline_report,
    save_classical_baseline_report,
    strongest_baseline,
)

DEFAULT_BACKTEST_DIR = Path("reports/backtests")
DEFAULT_COMPARISON_OUTPUT = (
    DEFAULT_BACKTEST_DIR / "classical_baseline_comparison.parquet"
)
DEFAULT_MARKDOWN_OUTPUT = DEFAULT_BACKTEST_DIR / "classical_baseline_comparison.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a classical baseline comparison report."
    )
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=DEFAULT_BACKTEST_DIR,
        help="Directory containing backtest metric parquet outputs.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT,
        help="Path to save comparison parquet.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Path to save Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classification_metrics = _read_parquet(
        args.backtest_dir / "classification_metrics.parquet"
    )
    ranking_metrics = _read_parquet(args.backtest_dir / "ranking_metrics.parquet")
    portfolio_risk_metrics = _read_parquet(
        args.backtest_dir / "portfolio_risk_metrics.parquet"
    )

    comparison = build_classical_baseline_comparison(
        classification_metrics=classification_metrics,
        ranking_metrics=ranking_metrics,
        portfolio_risk_metrics=portfolio_risk_metrics,
    )
    markdown = render_classical_baseline_report(comparison)
    save_classical_baseline_report(
        comparison=comparison,
        markdown=markdown,
        comparison_output=args.comparison_output,
        markdown_output=args.markdown_output,
    )

    print(f"Saved comparison table to {args.comparison_output}")
    print(f"Saved Markdown report to {args.markdown_output}")
    print(f"Strongest available baseline: {strongest_baseline(comparison)}")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest output not found: {path}")
    return pd.read_parquet(path)


if __name__ == "__main__":
    main()
