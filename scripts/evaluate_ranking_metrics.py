"""Evaluate ranking metrics from standard prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.backtest.ranking_metrics import (
    evaluate_ranking_metrics,
    load_prediction_tables,
    save_ranking_metrics,
)


DEFAULT_PREDICTION_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("data/processed/ranking_metrics.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model ranking quality by date and over time."
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
        help="Path to save ranking metrics parquet.",
    )
    parser.add_argument(
        "--return-column",
        default="forward_excess_return",
        help="Return column used to judge ranking quality.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Top and bottom fraction used for spread metrics.",
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
    metrics = evaluate_ranking_metrics(
        predictions,
        return_column=args.return_column,
        top_fraction=args.top_fraction,
    )
    save_ranking_metrics(metrics, args.output)

    print(f"Saved ranking metrics to {args.output}")
    print(f"Prediction files read: {len(prediction_paths)}")
    print(f"Prediction rows evaluated: {len(predictions)}")
    print("\nAggregate metrics:")
    print(metrics[metrics["scope"].isin(["split", "overall"])])


if __name__ == "__main__":
    main()
