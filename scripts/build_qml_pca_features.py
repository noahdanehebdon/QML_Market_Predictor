"""Build PCA-compressed feature vectors for QML experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_qml.backtest.splits import DEFAULT_SPLIT_OUTPUT_PATH
from market_qml.models.dataset import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    DEFAULT_TARGET_COLUMN,
)
from market_qml.qml.pca import (
    CLASSIFICATION_TARGET_COLUMN,
    DEFAULT_N_COMPONENTS,
    DEFAULT_QML_CLASSIFICATION_PCA_ARTIFACT_DIR,
    DEFAULT_QML_CLASSIFICATION_PCA_DIAGNOSTICS_PATH,
    DEFAULT_QML_CLASSIFICATION_PCA_FEATURE_PATH,
    DEFAULT_QML_REGRESSION_RANKING_PCA_ARTIFACT_DIR,
    DEFAULT_QML_REGRESSION_RANKING_PCA_DIAGNOSTICS_PATH,
    DEFAULT_QML_REGRESSION_RANKING_PCA_FEATURE_PATH,
    REGRESSION_RANKING_TARGET_COLUMN,
    build_grouped_qml_pca_features,
    build_qml_pca_features,
    save_qml_pca_artifacts,
    save_qml_pca_diagnostics,
    save_qml_pca_features,
)


@dataclass(frozen=True)
class QMLPCATask:
    """One task-specific QML compression output."""

    name: str
    target_column: str
    output_path: Path
    diagnostics_path: Path
    artifact_dir: Path


TASKS = {
    "classification": QMLPCATask(
        name="classification",
        target_column=CLASSIFICATION_TARGET_COLUMN,
        output_path=DEFAULT_QML_CLASSIFICATION_PCA_FEATURE_PATH,
        diagnostics_path=DEFAULT_QML_CLASSIFICATION_PCA_DIAGNOSTICS_PATH,
        artifact_dir=DEFAULT_QML_CLASSIFICATION_PCA_ARTIFACT_DIR,
    ),
    "regression_ranking": QMLPCATask(
        name="regression_ranking",
        target_column=REGRESSION_RANKING_TARGET_COLUMN,
        output_path=DEFAULT_QML_REGRESSION_RANKING_PCA_FEATURE_PATH,
        diagnostics_path=DEFAULT_QML_REGRESSION_RANKING_PCA_DIAGNOSTICS_PATH,
        artifact_dir=DEFAULT_QML_REGRESSION_RANKING_PCA_ARTIFACT_DIR,
    ),
}


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
        default=None,
        help=(
            "Optional single target override. When omitted, the script builds "
            "both classification and regression/ranking outputs."
        ),
    )
    parser.add_argument(
        "--task",
        choices=["both", "classification", "regression_ranking"],
        default="both",
        help="Task output to build when --target is not provided.",
    )
    parser.add_argument(
        "--compression",
        choices=["grouped", "global"],
        default="grouped",
        help="Compression strategy. Grouped PCA is the default QML input table.",
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
        default=None,
        help="Path to save a custom single-target PCA output.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Path to save custom single-target PCA diagnostics.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory to save custom single-target train-fitted PCA artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    splits = pd.read_parquet(args.splits)

    for task in _selected_tasks(args):
        result = _build_result(
            features=features,
            labels=labels,
            splits=splits,
            n_components=args.n_components,
            target_column=task.target_column,
            max_splits=args.max_splits,
            compression=args.compression,
        )

        save_qml_pca_features(result.features, task.output_path)
        save_qml_pca_diagnostics(result.diagnostics, task.diagnostics_path)
        artifact_paths = save_qml_pca_artifacts(result.artifacts, task.artifact_dir)

        print(f"Task: {task.name}")
        print(f"Target: {task.target_column}")
        print(f"Compression: {args.compression}")
        print(f"Saved QML PCA features to {task.output_path}")
        print(f"Saved PCA diagnostics to {task.diagnostics_path}")
        print(f"Saved PCA artifacts to {task.artifact_dir}")
        print(f"Artifact count: {len(artifact_paths)}")
        print(f"Rows: {len(result.features)}")
        print(f"Components: {args.n_components}")
        print(
            "Mean cumulative explained variance by group: "
            f"{result.diagnostics.groupby(['split_id', 'group'])['cumulative_explained_variance_ratio'].max().mean():.6f}"
        )


def _selected_tasks(args: argparse.Namespace) -> list[QMLPCATask]:
    if args.target:
        return [
            QMLPCATask(
                name="custom",
                target_column=args.target or DEFAULT_TARGET_COLUMN,
                output_path=args.output
                or Path("data/features/qml_pca_features.parquet"),
                diagnostics_path=args.diagnostics_output
                or Path("data/processed/qml_pca_explained_variance.parquet"),
                artifact_dir=args.artifact_dir or Path("artifacts/qml/pca"),
            )
        ]

    if any([args.output, args.diagnostics_output, args.artifact_dir]):
        raise ValueError(
            "--output, --diagnostics-output, and --artifact-dir are only supported "
            "with --target for a custom single-target run."
        )

    if args.task == "both":
        return [TASKS["classification"], TASKS["regression_ranking"]]
    return [TASKS[args.task]]


def _build_result(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    n_components: int,
    target_column: str,
    max_splits: int | None,
    compression: str,
):
    if compression == "grouped":
        return build_grouped_qml_pca_features(
            features=features,
            labels=labels,
            splits=splits,
            n_components=n_components,
            target_column=target_column,
            max_splits=max_splits,
        )
    return build_qml_pca_features(
        features=features,
        labels=labels,
        splits=splits,
        n_components=n_components,
        target_column=target_column,
        max_splits=max_splits,
    )


if __name__ == "__main__":
    main()
