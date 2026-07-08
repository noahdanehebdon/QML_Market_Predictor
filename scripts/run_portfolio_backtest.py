"""Run simple portfolio backtests from standard prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.backtest.portfolio import (
    load_prediction_tables,
    run_portfolio_backtest,
    save_portfolio_returns,
    save_portfolio_risk_metrics,
    summarize_portfolio_risk,
)


DEFAULT_PREDICTION_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("data/processed/portfolio_backtest.parquet")
DEFAULT_RISK_OUTPUT_PATH = Path("data/processed/portfolio_risk_metrics.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an equal-weight top-ranked portfolio backtest."
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help="Directory containing prediction parquet files.",
    )
    parser.add_argument(
        "--pattern",
        default="predictions_*.parquet",
        help="Glob pattern for prediction parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save portfolio backtest parquet.",
    )
    parser.add_argument(
        "--risk-output",
        type=Path,
        default=DEFAULT_RISK_OUTPUT_PATH,
        help="Path to save portfolio risk metrics parquet.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top-ranked names to select each date.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction of top-ranked names to select when top-k is not set.",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in basis points applied to turnover.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=5,
        help="Number of prediction dates between portfolio rebalances.",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualization periods used for volatility and Sharpe ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = sorted(args.prediction_dir.glob(args.pattern))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files found in {args.prediction_dir} matching {args.pattern}."
        )

    predictions = load_prediction_tables(prediction_paths)
    portfolio_returns = run_portfolio_backtest(
        predictions,
        top_k=args.top_k,
        top_fraction=args.top_fraction,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_frequency=args.rebalance_frequency,
    )
    risk_metrics = summarize_portfolio_risk(
        portfolio_returns,
        periods_per_year=args.periods_per_year,
    )
    save_portfolio_returns(portfolio_returns, args.output)
    save_portfolio_risk_metrics(risk_metrics, args.risk_output)

    print(f"Saved portfolio backtest to {args.output}")
    print(f"Saved portfolio risk metrics to {args.risk_output}")
    print(f"Prediction files read: {len(prediction_paths)}")
    print(f"Return rows: {len(portfolio_returns)}")
    print("\nFinal cumulative returns:")
    print(portfolio_returns.groupby(["model_name", "split_id"]).tail(1))
    print("\nRisk metrics:")
    print(risk_metrics)


if __name__ == "__main__":
    main()
