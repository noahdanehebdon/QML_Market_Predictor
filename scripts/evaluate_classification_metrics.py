"""Evaluate classification metrics from standard prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from market_qml.backtest.classification_metrics import (
    evaluate_classification_metrics,
    load_prediction_tables,
    save_classification_metrics,
)


DEFAULT_PREDICTION_DIR = Path("data/processed")
DEFAULT_OUTPUT_PATH = Path("data/processed/classification_metrics.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary classification metrics by split and overall."
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
        help="Path to save classification metrics parquet.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for accuracy, precision, and recall.",
    )
    parser.add_argument(
        "--include-non-binary",
        action="store_true",
        help="Error on non-binary targets instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = sorted(args.prediction_dir.glob(args.pattern))
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction files found in {args.prediction_dir} matching {args.pattern}."
        )

    predictions = load_prediction_tables(
        prediction_paths,
        skip_non_binary=not args.include_non_binary,
    )
    metrics = evaluate_classification_metrics(predictions, threshold=args.threshold)
    save_classification_metrics(metrics, args.output)

    print(f"Saved classification metrics to {args.output}")
    print(f"Prediction files read: {len(prediction_paths)}")
    print(f"Prediction rows evaluated: {len(predictions)}")
    print("\nMetrics:")
    print(metrics)


if __name__ == "__main__":
    main()
