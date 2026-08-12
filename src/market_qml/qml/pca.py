"""Leakage-safe PCA compression for QML input features."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from market_qml.models.dataset import (
    DEFAULT_TARGET_COLUMN,
    build_train_validation_datasets,
)
from market_qml.models.preprocessing import (
    PreprocessedDataset,
    fit_transform_train_validation,
)
from market_qml.qml.pca_io import load_artifact, save_artifacts, save_frame
from market_qml.qml.pca_types import (
    GroupedPCAArtifact,
    PCAArtifact,
    QMLPCACompressionResult,
)

CLASSIFICATION_TARGET_COLUMN = "outperform_spy_5d"
REGRESSION_RANKING_TARGET_COLUMN = "forward_excess_return_5d"
DEFAULT_QML_CLASSIFICATION_PCA_FEATURE_PATH = Path(
    "data/features/qml_classification_grouped_pca_features.parquet"
)
DEFAULT_QML_CLASSIFICATION_PCA_DIAGNOSTICS_PATH = Path(
    "data/processed/qml_classification_grouped_pca_explained_variance.parquet"
)
DEFAULT_QML_CLASSIFICATION_PCA_ARTIFACT_DIR = Path("artifacts/qml/pca_classification")
DEFAULT_QML_REGRESSION_RANKING_PCA_FEATURE_PATH = Path(
    "data/features/qml_regression_ranking_grouped_pca_features.parquet"
)
DEFAULT_QML_REGRESSION_RANKING_PCA_DIAGNOSTICS_PATH = Path(
    "data/processed/qml_regression_ranking_grouped_pca_explained_variance.parquet"
)
DEFAULT_QML_REGRESSION_RANKING_PCA_ARTIFACT_DIR = Path(
    "artifacts/qml/pca_regression_ranking"
)
DEFAULT_QML_PCA_FEATURE_PATH = DEFAULT_QML_CLASSIFICATION_PCA_FEATURE_PATH
DEFAULT_QML_PCA_DIAGNOSTICS_PATH = DEFAULT_QML_CLASSIFICATION_PCA_DIAGNOSTICS_PATH
DEFAULT_QML_PCA_ARTIFACT_DIR = DEFAULT_QML_CLASSIFICATION_PCA_ARTIFACT_DIR
DEFAULT_N_COMPONENTS = 16

FEATURE_GROUP_COMPONENT_TARGETS = OrderedDict(
    [
        ("raw_price", 2),
        ("returns_momentum", 3),
        ("volatility", 2),
        ("volume_liquidity", 2),
        ("benchmark_relative", 3),
        ("macro", 2),
        ("fundamentals", 1),
        ("other", 1),
    ]
)


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
                group_name="global",
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


def build_grouped_qml_pca_features(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    splits: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    target_column: str = DEFAULT_TARGET_COLUMN,
    max_splits: int | None = None,
) -> QMLPCACompressionResult:
    """Build split-aware grouped PCA QML feature rows."""
    _validate_n_components(n_components)
    selected_splits = _selected_splits(splits, max_splits=max_splits)

    feature_frames = []
    diagnostic_frames = []
    artifacts: dict[int, GroupedPCAArtifact] = {}

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
        group_columns = infer_feature_groups(preprocessed.train.X.columns)
        component_counts = allocate_group_components(
            group_columns=group_columns,
            n_components=n_components,
            train_rows=len(preprocessed.train.X),
        )
        pcas = fit_grouped_pca(
            preprocessed.train.X,
            group_columns=group_columns,
            component_counts=component_counts,
        )

        train_features = transform_grouped_pca_dataset(
            preprocessed.train,
            pcas=pcas,
            group_columns=group_columns,
            split_id=split_id,
            sample_role="train",
        )
        validation_features = transform_grouped_pca_dataset(
            preprocessed.validation,
            pcas=pcas,
            group_columns=group_columns,
            split_id=split_id,
            sample_role="validation",
        )
        feature_frames.extend([train_features, validation_features])
        diagnostic_frames.append(
            grouped_pca_diagnostics(
                pcas=pcas,
                group_columns=group_columns,
                split_id=split_id,
                train_rows=len(preprocessed.train.X),
                validation_rows=len(preprocessed.validation.X),
            )
        )
        artifacts[split_id] = GroupedPCAArtifact(
            split_id=split_id,
            target_column=target_column,
            feature_columns=list(preprocessed.train.X.columns),
            group_columns={
                group_name: list(columns)
                for group_name, columns in group_columns.items()
                if group_name in pcas
            },
            component_columns=[
                column
                for group_name, pca in pcas.items()
                for column in _group_component_columns(group_name, pca.n_components_)
            ],
            preprocessor=preprocessed.preprocessor,
            pcas=pcas,
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


def fit_grouped_pca(
    train_X: pd.DataFrame,
    *,
    group_columns: OrderedDict[str, list[str]],
    component_counts: dict[str, int],
) -> dict[str, PCA]:
    """Fit one PCA per feature group on training data only."""
    if train_X.empty:
        raise ValueError("Cannot fit grouped PCA on an empty training matrix.")

    pcas = {}
    for group_name, columns in group_columns.items():
        n_components = component_counts.get(group_name, 0)
        if n_components <= 0:
            continue
        pcas[group_name] = fit_pca(train_X[columns], n_components=n_components)
    if not pcas:
        raise ValueError("Grouped PCA did not select any feature groups.")
    return pcas


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

    return result.sort_values(
        ["split_id", "sample_role", "symbol", "date"]
    ).reset_index(drop=True)


def transform_grouped_pca_dataset(
    dataset: PreprocessedDataset,
    *,
    pcas: dict[str, PCA],
    group_columns: OrderedDict[str, list[str]],
    split_id: int,
    sample_role: str,
) -> pd.DataFrame:
    """Transform one preprocessed dataset into grouped PCA component rows."""
    component_frames = []
    for group_name, pca in pcas.items():
        components = pca.transform(dataset.X[group_columns[group_name]])
        component_frames.append(
            pd.DataFrame(
                components,
                columns=_group_component_columns(group_name, pca.n_components_),
                index=dataset.X.index,
            )
        )

    result = dataset.metadata[["symbol", "date"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["split_id"] = split_id
    result["sample_role"] = sample_role
    result["target"] = pd.to_numeric(dataset.y, errors="coerce").to_numpy()
    result = pd.concat(
        [result.reset_index(drop=True)]
        + [frame.reset_index(drop=True) for frame in component_frames],
        axis=1,
    )

    if result["date"].isna().any():
        raise ValueError("PCA output metadata contains invalid dates.")
    if result["target"].isna().any():
        raise ValueError("PCA output contains missing or non-numeric targets.")

    return result.sort_values(
        ["split_id", "sample_role", "symbol", "date"]
    ).reset_index(drop=True)


def pca_diagnostics(
    *,
    pca: PCA,
    group_name: str,
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
                "group": group_name,
                "component": _group_component_columns(group_name, pca.n_components_)[
                    index
                ],
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


def grouped_pca_diagnostics(
    *,
    pcas: dict[str, PCA],
    group_columns: OrderedDict[str, list[str]],
    split_id: int,
    train_rows: int,
    validation_rows: int,
) -> pd.DataFrame:
    """Build explained-variance diagnostics for grouped PCA artifacts."""
    frames = []
    for group_name, pca in pcas.items():
        frames.append(
            pca_diagnostics(
                pca=pca,
                group_name=group_name,
                split_id=split_id,
                n_original_features=len(group_columns[group_name]),
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
        )
    return pd.concat(frames, ignore_index=True)


def save_qml_pca_features(
    pca_features: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_PCA_FEATURE_PATH,
) -> None:
    """Save PCA-compressed QML features to parquet."""
    save_frame(pca_features, output_path)


def save_qml_pca_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_PCA_DIAGNOSTICS_PATH,
) -> None:
    """Save PCA explained-variance diagnostics to parquet."""
    save_frame(diagnostics, output_path)


def save_qml_pca_artifacts(
    artifacts: dict[int, PCAArtifact | GroupedPCAArtifact],
    artifact_dir: str | Path = DEFAULT_QML_PCA_ARTIFACT_DIR,
) -> list[Path]:
    """Save train-fitted PCA artifacts, one pickle per split."""
    return save_artifacts(artifacts, artifact_dir)


def load_qml_pca_artifact(path: str | Path) -> PCAArtifact:
    """Load one saved PCA artifact."""
    return load_artifact(path)


def infer_feature_groups(columns: pd.Index | list[str]) -> OrderedDict[str, list[str]]:
    """Group feature columns by economic family for grouped compression."""
    groups: OrderedDict[str, list[str]] = OrderedDict(
        (group_name, []) for group_name in FEATURE_GROUP_COMPONENT_TARGETS
    )
    for column in columns:
        group_name = _feature_group(str(column))
        groups[group_name].append(str(column))
    return groups


def allocate_group_components(
    *,
    group_columns: OrderedDict[str, list[str]],
    n_components: int,
    train_rows: int,
) -> dict[str, int]:
    """Allocate a fixed component budget across non-empty feature groups."""
    _validate_n_components(n_components)
    if train_rows <= 0:
        raise ValueError("train_rows must be positive.")

    capacities = OrderedDict(
        (
            group_name,
            min(len(columns), train_rows),
        )
        for group_name, columns in group_columns.items()
        if columns
    )
    total_capacity = sum(capacities.values())
    if n_components > total_capacity:
        raise ValueError("n_components cannot exceed grouped feature capacity.")

    allocations = {
        group_name: min(
            FEATURE_GROUP_COMPONENT_TARGETS.get(group_name, 1),
            capacity,
        )
        for group_name, capacity in capacities.items()
    }
    while sum(allocations.values()) > n_components:
        group_name = max(allocations, key=lambda name: allocations[name])
        allocations[group_name] -= 1
        if allocations[group_name] == 0:
            del allocations[group_name]

    while sum(allocations.values()) < n_components:
        changed = False
        for group_name, capacity in capacities.items():
            current = allocations.get(group_name, 0)
            if current < capacity:
                allocations[group_name] = current + 1
                changed = True
                if sum(allocations.values()) == n_components:
                    break
        if not changed:
            break

    return {group_name: count for group_name, count in allocations.items() if count > 0}


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


def _group_component_columns(group_name: str, n_components: int) -> list[str]:
    return [f"{group_name}_pca_{index:02d}" for index in range(n_components)]


def _feature_group(column: str) -> str:
    lowered = column.lower()
    if lowered in {"open", "high", "low", "close", "vwap", "trade_count"}:
        return "raw_price"
    if "volume" in lowered or "liquid" in lowered:
        return "volume_liquidity"
    if "vol" in lowered or "realized" in lowered:
        return "volatility"
    if lowered.startswith("relative_") or "_vs_" in lowered or "benchmark" in lowered:
        return "benchmark_relative"
    if any(
        marker in lowered
        for marker in (
            "return",
            "momentum",
            "moving_average",
            "ma_",
            "rsi",
            "macd",
        )
    ):
        return "returns_momentum"
    if any(
        marker in lowered
        for marker in (
            "treasury",
            "fed_funds",
            "cpi",
            "unemployment",
            "industrial_production",
            "inflation",
            "macro",
        )
    ):
        return "macro"
    if any(
        marker in lowered
        for marker in (
            "sec_",
            "filing",
            "assets",
            "liabilities",
            "revenue",
            "income",
            "debt",
            "equity",
            "cash",
            "eps",
            "shares",
        )
    ):
        return "fundamentals"
    return "other"


def _validate_n_components(n_components: int) -> None:
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
