"""Run a reproducible QCNN initialization, learning-rate, and sample-size grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qcnn_stability import (
    evaluate_qcnn_stability,
    save_qcnn_stability_result,
)


DEFAULT_SAMPLE_PATH = Path("data/features/qml_sample_grouped_smoke.parquet")
DEFAULT_LABEL_PATH = Path("data/labels/forward_return_labels.parquet")
DEFAULT_OUTPUT_DIR = Path("reports/qcnn_stability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QCNN training stability.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument(
        "--initialization-scales",
        type=float,
        nargs="+",
        default=[0.01, 0.1],
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
    )
    parser.add_argument(
        "--train-sample-sizes",
        type=int,
        nargs="+",
        default=[128, 512],
    )
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = _attach_returns(
        pd.read_parquet(args.sample),
        pd.read_parquet(args.labels),
    )
    data = build_qml_train_validation(
        sample,
        split_id=args.split_id,
        feature_columns=_feature_columns(sample),
    )
    result = evaluate_qcnn_stability(
        data,
        initialization_scales=args.initialization_scales,
        learning_rates=args.learning_rates,
        train_sample_sizes=args.train_sample_sizes,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )
    paths = save_qcnn_stability_result(result, output_dir=args.output_dir)
    print(f"Selected stable QCNN configuration: {result.best_config['config_id']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


def _attach_returns(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "date",
        "forward_return_5d",
        "forward_excess_return_5d",
    ]
    missing = set(columns) - set(labels.columns)
    if missing:
        raise ValueError(
            "Label table is missing required QCNN return columns: "
            + ", ".join(sorted(missing))
        )
    left = sample.copy()
    right = labels[columns].copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    return left.merge(
        right,
        on=["symbol", "date"],
        how="left",
        validate="many_to_one",
    )


def _feature_columns(sample: pd.DataFrame) -> list[str]:
    columns = sorted(
        column
        for column in sample.columns
        if column.startswith("pca_") or "_pca_" in column
    )
    if len(columns) < 8:
        raise ValueError("QML sample must contain at least 8 PCA columns.")
    return columns[:8]


if __name__ == "__main__":
    main()
