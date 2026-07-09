"""Build PCA-compressed feature vectors for QML experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.backtest.splits import DEFAULT_SPLIT_OUTPUT_PATH
from market_qml.models.dataset import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_TARGET_COLUMN,
)
from market_qml.qml.pca import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_QML_PCA_ARTIFACT_DIR,
    DEFAULT_QML_PCA_DIAGNOSTICS_PATH,
    DEFAULT_QML_PCA_FEATURE_PATH,
    build_qml_pca_features,
    save_qml_pca_artifacts,
    save_qml_pca_diagnostics,
    save_qml_pca_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe PCA-compressed inputs for QML experiments."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to canonical feature table parquet.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help="Path to forward return label table parquet.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT_PATH,
        help="Path to walk-forward split metadata parquet.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET_COLUMN,
        help="Target column from the label table.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Number of PCA components, usually matching the qubit count.",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Optional cap on number of splits for quick smoke runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QML_PCA_FEATURE_PATH,
        help="Path to save PCA-compressed QML feature rows.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=DEFAULT_QML_PCA_DIAGNOSTICS_PATH,
        help="Path to save PCA explained-variance diagnostics.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_QML_PCA_ARTIFACT_DIR,
        help="Directory to save train-fitted PCA artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_qml_pca_features(
        features=pd.read_parquet(args.features),
        labels=pd.read_parquet(args.labels),
        splits=pd.read_parquet(args.splits),
        n_components=args.n_components,
        target_column=args.target,
        max_splits=args.max_splits,
    )

    save_qml_pca_features(result.features, args.output)
    save_qml_pca_diagnostics(result.diagnostics, args.diagnostics_output)
    artifact_paths = save_qml_pca_artifacts(result.artifacts, args.artifact_dir)

    print(f"Saved QML PCA features to {args.output}")
    print(f"Saved PCA diagnostics to {args.diagnostics_output}")
    print(f"Saved PCA artifacts to {args.artifact_dir}")
    print(f"Artifact count: {len(artifact_paths)}")
    print(f"Rows: {len(result.features)}")
    print(f"Components: {args.n_components}")
    print(
        "Mean cumulative explained variance: "
        f"{result.diagnostics.groupby('split_id')['cumulative_explained_variance_ratio'].max().mean():.6f}"
    )


if __name__ == "__main__":
    main()
