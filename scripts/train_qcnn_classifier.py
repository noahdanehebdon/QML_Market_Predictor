"""Train the eight-qubit QCNN classifier on a reduced QML sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qcnn import save_qcnn_result, train_qcnn

DEFAULT_SAMPLE_PATH = Path("data/features/qml_sample_grouped_smoke.parquet")
DEFAULT_LABEL_PATH = Path("data/labels/forward_return_labels.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/qml/qcnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the eight-qubit QCNN classifier."
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--perturbation", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--initialization-scale", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = _attach_return_metadata(
        pd.read_parquet(args.sample),
        pd.read_parquet(args.labels),
    )
    feature_columns = _qml_feature_columns(sample)
    data = build_qml_train_validation(
        sample,
        split_id=args.split_id,
        feature_columns=feature_columns,
    )
    result = train_qcnn(
        data,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        batch_size=args.batch_size,
        l2=args.l2,
        initialization_scale=args.initialization_scale,
        random_state=args.random_state,
    )
    paths = save_qcnn_result(result, output_dir=args.output_dir)
    print(
        f"Trained QCNN on {len(data.train.y)} rows and predicted "
        f"{len(data.validation.y)} validation rows."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def _attach_return_metadata(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
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


def _qml_feature_columns(sample: pd.DataFrame) -> list[str]:
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
