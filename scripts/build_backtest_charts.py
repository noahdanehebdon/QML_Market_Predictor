"""Generate reproducible backtest and model-comparison charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.reporting.backtest_charts import (
    generate_backtest_charts,
    render_chart_report,
    save_chart_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate backtest result charts.")
    parser.add_argument(
        "--portfolio-returns",
        type=Path,
        default=Path("reports/qml_comparison/portfolio_returns.parquet"),
    )
    parser.add_argument(
        "--model-summary",
        type=Path,
        default=Path("reports/model_comparison.csv"),
    )
    parser.add_argument(
        "--regime-metrics",
        type=Path,
        default=Path("reports/qml_regimes/regime_metrics.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--report-output", type=Path, default=Path("reports/backtest_charts.md")
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = [args.portfolio_returns, args.model_summary, args.regime_metrics]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required chart inputs are missing: " + ", ".join(missing)
        )

    paths = generate_backtest_charts(
        portfolio_returns=pd.read_parquet(args.portfolio_returns),
        model_summary=pd.read_csv(args.model_summary),
        regime_metrics=pd.read_parquet(args.regime_metrics),
        output_dir=args.output_dir,
        rolling_window=args.rolling_window,
    )
    markdown = render_chart_report(paths, report_path=args.report_output)
    save_chart_report(markdown, args.report_output)
    print(f"Saved {len(paths)} figures to {args.output_dir}")
    print(f"Saved chart report to {args.report_output}")


if __name__ == "__main__":
    main()
