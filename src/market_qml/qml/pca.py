"""Leakage-safe PCA compression for QML input features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.decomposition import PCA

from market_qml.models.dataset import (
    DEFAULT_TARGET_COLUMN,
    build_train_validation_datasets,
)
from market_qml.models.preprocessing import (
    FittedPreprocessor,
    PreprocessedDataset,
    fit_transform_train_validation,
)


DEFAULT_QML_PCA_FEATURE_PATH = Path("data/features/qml_pca_features.parquet")
DEFAULT_QML_PCA_DIAGNOSTICS_PATH = Path("data/processed/qml_pca_explained_variance.parquet")
DEFAULT_QML_PCA_ARTIFACT_DIR = Path("artifacts/qml/pca")
DEFAULT_N_COMPONENTS = 16


@dataclass(frozen=True)
class PCAArtifact:
    """Train-fitted preprocessing and PCA state for one split."""

    split_id: int
    target_column: str
    feature_columns: list[str]
    preprocessor: FittedPreprocessor
    pca: PCA


@dataclass(frozen=True)
class QMLPCACompressionResult:
    """Compressed QML feature rows, diagnostics, and fitted PCA artifacts."""

    features: pd.DataFrame
    diagnostics: pd.DataFrame
    artifacts: dict[int, PCAArtifact]


def build_qml_pca_features(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    target_column: str = DEFAULT_TARGET_COLUMN,
    max_splits: int | None = None,
) -> QMLPCACompressionResult:
    """Build split-aware PCA-compressed train/validation QML feature rows."""
    _validate_n_components(n_components)
    selected_splits = _selected_splits(splits, max_splits=max_splits)

    feature_frames = []
    diagnostic_frames = []
    artifacts: dict[int, PCAArtifact] = {}

    for split in selected_splits.itertuples(index=False):
        datasets = build_train_validation_datasets(
            features=features,
            labels=labels,
            target_column=target_column,
            train_start_date=split.train_start_date,
            train_end_date=split.train_end_date,
            validation_start_date=split.validation_start_date,
            validation_end_date=split.validation_end_date,
        )
        preprocessed = fit_transform_train_validation(datasets)
        split_id = int(split.split_id)
        pca = fit_pca(preprocessed.train.X, n_components=n_components)

        train_features = transform_pca_dataset(
            preprocessed.train,
            pca=pca,
            split_id=split_id,
            sample_role="train",
        )
        validation_features = transform_pca_dataset(
            preprocessed.validation,
            pca=pca,
            split_id=split_id,
            sample_role="validation",
        )
        feature_frames.extend([train_features, validation_features])
        diagnostic_frames.append(
            pca_diagnostics(
                pca=pca,
                split_id=split_id,
                n_original_features=len(preprocessed.train.X.columns),
                train_rows=len(preprocessed.train.X),
                validation_rows=len(preprocessed.validation.X),
            )
        )
        artifacts[split_id] = PCAArtifact(
            split_id=split_id,
            target_column=target_column,
            feature_columns=list(preprocessed.train.X.columns),
            preprocessor=preprocessed.preprocessor,
            pca=pca,
        )

    return QMLPCACompressionResult(
        features=pd.concat(feature_frames, ignore_index=True),
        diagnostics=pd.concat(diagnostic_frames, ignore_index=True),
        artifacts=artifacts,
    )


def fit_pca(train_X: pd.DataFrame, *, n_components: int) -> PCA:
    """Fit PCA on training features only."""
    _validate_n_components(n_components)
    if train_X.empty:
        raise ValueError("Cannot fit PCA on an empty training matrix.")
    if n_components > len(train_X.columns):
        raise ValueError("n_components cannot exceed the number of feature columns.")
    if n_components > len(train_X):
        raise ValueError("n_components cannot exceed the number of training rows.")

    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(train_X)
    return pca


def transform_pca_dataset(
    dataset: PreprocessedDataset,
    *,
    pca: PCA,
    split_id: int,
    sample_role: str,
) -> pd.DataFrame:
    """Transform one preprocessed dataset into PCA component rows."""
    components = pca.transform(dataset.X)
    component_columns = _component_columns(pca.n_components_)
    component_frame = pd.DataFrame(
        components,
        columns=component_columns,
        index=dataset.X.index,
    )

    result = dataset.metadata[["symbol", "date"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["split_id"] = split_id
    result["sample_role"] = sample_role
    result["target"] = pd.to_numeric(dataset.y, errors="coerce").to_numpy()
    result = pd.concat(
        [result.reset_index(drop=True), component_frame.reset_index(drop=True)],
        axis=1,
    )

    if result["date"].isna().any():
        raise ValueError("PCA output metadata contains invalid dates.")
    if result["target"].isna().any():
        raise ValueError("PCA output contains missing or non-numeric targets.")

    return result.sort_values(["split_id", "sample_role", "symbol", "date"]).reset_index(
        drop=True
    )


def pca_diagnostics(
    *,
    pca: PCA,
    split_id: int,
    n_original_features: int,
    train_rows: int,
    validation_rows: int,
) -> pd.DataFrame:
    """Build explained-variance diagnostics for one fitted PCA artifact."""
    rows = []
    cumulative = 0.0
    for index, ratio in enumerate(pca.explained_variance_ratio_):
        cumulative += float(ratio)
        rows.append(
            {
                "split_id": split_id,
                "component": f"pca_{index:02d}",
                "component_index": index,
                "n_original_features": n_original_features,
                "n_components": int(pca.n_components_),
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "explained_variance": float(pca.explained_variance_[index]),
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance_ratio": cumulative,
            }
        )
    return pd.DataFrame(rows)


def save_qml_pca_features(
    pca_features: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_PCA_FEATURE_PATH,
) -> None:
    """Save PCA-compressed QML features to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pca_features.to_parquet(output_path, index=False)


def save_qml_pca_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_PCA_DIAGNOSTICS_PATH,
) -> None:
    """Save PCA explained-variance diagnostics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(output_path, index=False)


def save_qml_pca_artifacts(
    artifacts: dict[int, PCAArtifact],
    artifact_dir: str | Path = DEFAULT_QML_PCA_ARTIFACT_DIR,
) -> list[Path]:
    """Save train-fitted PCA artifacts, one pickle per split."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for split_id, artifact in sorted(artifacts.items()):
        path = artifact_dir / f"pca_split_{split_id:03d}.pkl"
        with path.open("wb") as f:
            pickle.dump(artifact, f)
        paths.append(path)
    return paths


def load_qml_pca_artifact(path: str | Path) -> PCAArtifact:
    """Load one saved PCA artifact."""
    path = Path(path)
    with path.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, PCAArtifact):
        raise TypeError(f"Unexpected PCA artifact type: {type(artifact)!r}")
    return artifact


def _selected_splits(splits: pd.DataFrame, *, max_splits: int | None) -> pd.DataFrame:
    if splits.empty:
        raise ValueError("Walk-forward split table is empty.")
    if max_splits is not None and max_splits <= 0:
        raise ValueError("max_splits must be positive when provided.")

    selected = splits.sort_values("split_id").reset_index(drop=True)
    if max_splits is not None:
        selected = selected.head(max_splits)
    return selected


def _component_columns(n_components: int) -> list[str]:
    return [f"pca_{index:02d}" for index in range(n_components)]


def _validate_n_components(n_components: int) -> None:
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
