"""Build quantum feature-map circuit outputs for one sampled QML split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.feature_map import (
    QuantumFeatureMapConfig,
    QuantumKernelFeatureMap,
    save_feature_map_split,
)
from market_qml.qml.interface import build_qml_train_validation

DEFAULT_SAMPLE_PATH = Path(
    "data/features/qml_classification_grouped_pca_features.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/qml/quantum_feature_map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simulator-backed quantum kernel feature-map states."
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
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
    feature_map = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(
            n_qubits=args.n_qubits,
            repetitions=args.repetitions,
        )
    )
    result = feature_map.transform_train_validation(data)
    paths = save_feature_map_split(result, output_dir=args.output_dir)
    print(
        f"Built feature-map states for {len(result.train.states)} training and "
        f"{len(result.validation.states)} validation rows."
    )
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
