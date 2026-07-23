"""Train the QSVM baseline on a reduced PCA-compressed QML sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qsvm import save_qsvm_result, train_qsvm

DEFAULT_SAMPLE_PATH = Path("data/features/qml_sample_grouped_smoke.parquet")
DEFAULT_LABEL_PATH = Path("data/labels/forward_return_labels.parquet")
DEFAULT_OUTPUT_DIR = Path("artifacts/qml/qsvm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a quantum fidelity-kernel SVM on a reduced sample."
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = _attach_return_metadata(
        pd.read_parquet(args.sample),
        pd.read_parquet(args.labels),
    )
    feature_columns = _qml_feature_columns(sample, n_qubits=args.n_qubits)
    data = build_qml_train_validation(
        sample,
        split_id=args.split_id,
        feature_columns=feature_columns,
    )
    result = train_qsvm(
        data,
        C=args.C,
        n_qubits=args.n_qubits,
        repetitions=args.repetitions,
        random_state=args.random_state,
    )
    paths = save_qsvm_result(
        result,
        model_path=args.output_dir / "qsvm.pkl",
        prediction_path=args.output_dir / "predictions.parquet",
        diagnostics_path=args.output_dir / "kernel_diagnostics.parquet",
        kernel_path=args.output_dir / "kernel_matrices.npz",
    )
    print(
        f"Trained QSVM on {result.train_kernel.shape[0]} rows and predicted "
        f"{result.validation_kernel.shape[0]} validation rows."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


def _attach_return_metadata(sample: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol",
        "date",
        "forward_return_5d",
        "forward_excess_return_5d",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            "Label table is missing required QSVM return columns: "
            + ", ".join(sorted(missing))
        )
    left = sample.copy()
    right = labels[list(required)].copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce").dt.normalize()
    right["date"] = pd.to_datetime(right["date"], errors="coerce").dt.normalize()
    result = left.merge(
        right, on=["symbol", "date"], how="left", validate="many_to_one"
    )
    return result


def _qml_feature_columns(sample: pd.DataFrame, *, n_qubits: int) -> list[str]:
    columns = sorted(
        column
        for column in sample.columns
        if column.startswith("pca_") or "_pca_" in column
    )
    if len(columns) < n_qubits:
        raise ValueError(
            f"QML sample contains {len(columns)} PCA columns; {n_qubits} are required."
        )
    return columns[:n_qubits]


if __name__ == "__main__":
    main()
