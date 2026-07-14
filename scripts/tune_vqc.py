"""Tune VQC depth, learning rate, and optimizer on one sampled split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.tuning import save_vqc_tuning_result, tune_vqc


DEFAULT_SAMPLE_PATH = Path(
    "data/features/qml_classification_grouped_pca_features.parquet"
)
DEFAULT_OUTPUT_DIR = Path("reports/vqc_tuning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune VQC architecture and optimizer.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--ansatz-depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=["spsa", "finite_difference"],
        default=["spsa", "finite_difference"],
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overfit-gap-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = pd.read_parquet(args.sample)
    feature_columns = _qml_feature_columns(sample, n_qubits=args.n_qubits)
    data = build_qml_train_validation(
        sample,
        split_id=args.split_id,
        feature_columns=feature_columns,
    )
    result = tune_vqc(
        data,
        ansatz_depths=args.ansatz_depths,
        learning_rates=args.learning_rates,
        optimizers=args.optimizers,
        max_iter=args.max_iter,
        n_qubits=args.n_qubits,
        batch_size=args.batch_size,
        random_state=args.random_state,
        overfit_gap_threshold=args.overfit_gap_threshold,
    )
    paths = save_vqc_tuning_result(result, output_dir=args.output_dir)
    print(f"Best VQC configuration: {result.best_config['config_id']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


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
