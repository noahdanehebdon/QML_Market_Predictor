"""Modeling dataset construction from canonical features and labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from market_qml.features.canonical import build_canonical_features

DEFAULT_FEATURE_PATH = Path("data/features/feature_table.parquet")
DEFAULT_LABEL_PATH = Path("data/labels/forward_return_labels.parquet")
DEFAULT_TARGET_COLUMN = "outperform_spy_5d"
KEY_COLUMNS = ["symbol", "date"]


@dataclass(frozen=True)
class ModelingDataset:
    """Feature matrix, target vector, and row metadata for model training."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class TrainValidationDatasets:
    """Train and validation modeling datasets built with identical columns."""

    train: ModelingDataset
    validation: ModelingDataset


def build_modeling_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    universe_membership: pd.DataFrame | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    metadata_columns: list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    max_missing_feature_fraction: float | None = None,
) -> ModelingDataset:
    """Join canonical features and labels into X, y, and metadata."""
    feature_table = build_canonical_features(features)
    if universe_membership is not None:
        feature_table = _apply_universe_membership(feature_table, universe_membership)
    label_table = _prepare_labels(labels=labels, target_column=target_column)

    merged = feature_table.merge(
        label_table,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    label_columns = [
        column for column in label_table.columns if column not in KEY_COLUMNS
    ]
    merged = _filter_dates(merged, start_date=start_date, end_date=end_date)
    merged = merged.dropna(subset=[target_column]).reset_index(drop=True)

    if feature_columns is None:
        feature_columns = _infer_feature_columns(
            merged,
            excluded_columns=set(KEY_COLUMNS + label_columns),
        )
    else:
        _validate_requested_columns(merged, feature_columns, "feature_columns")

    if not feature_columns:
        raise ValueError("No feature columns were selected.")

    if metadata_columns is None:
        metadata_columns = _infer_metadata_columns(merged, feature_columns)
    else:
        _validate_requested_columns(merged, metadata_columns, "metadata_columns")

    if max_missing_feature_fraction is not None:
        _validate_missing_fraction(max_missing_feature_fraction)
        max_missing_count = int(len(feature_columns) * max_missing_feature_fraction)
        missing_counts = merged[feature_columns].isna().sum(axis=1)
        merged = merged[missing_counts <= max_missing_count].reset_index(drop=True)

    X = merged[feature_columns].copy()
    y = merged[target_column].copy()
    metadata = merged[metadata_columns].copy()

    return ModelingDataset(X=X, y=y, metadata=metadata)


def _apply_universe_membership(
    features: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Keep only effective-date universe members and attach PIT classifications."""
    required = set(KEY_COLUMNS + ["is_member"])
    missing = required - set(membership)
    if missing:
        raise ValueError(
            "Universe membership is missing required columns: "
            + ", ".join(sorted(missing))
        )
    selected = membership.copy()
    selected["symbol"] = selected["symbol"].astype(str).str.upper()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.normalize()
    if selected["date"].isna().any():
        raise ValueError("Universe membership contains invalid dates.")
    if selected.duplicated(KEY_COLUMNS).any():
        raise ValueError("Universe membership contains duplicate symbol/date rows.")
    metadata = [
        column for column in ["sector", "industry", "size_bucket"] if column in selected
    ]
    selected = selected.loc[selected["is_member"].eq(True), KEY_COLUMNS + metadata]
    return features.merge(selected, on=KEY_COLUMNS, how="inner", validate="one_to_one")


def load_modeling_dataset(
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
    label_path: str | Path = DEFAULT_LABEL_PATH,
    **kwargs,
) -> ModelingDataset:
    """Load feature/label parquet files and construct a modeling dataset."""
    feature_path = Path(feature_path)
    label_path = Path(label_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label table not found: {label_path}")

    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)

    return build_modeling_dataset(features=features, labels=labels, **kwargs)


def build_train_validation_datasets(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    train_start_date: str | pd.Timestamp | None = None,
    train_end_date: str | pd.Timestamp | None = None,
    validation_start_date: str | pd.Timestamp | None = None,
    validation_end_date: str | pd.Timestamp | None = None,
    **kwargs,
) -> TrainValidationDatasets:
    """Build train and validation datasets using date filters."""
    dataset_kwargs = dict(kwargs)
    requested_feature_columns = dataset_kwargs.get("feature_columns")
    train = build_modeling_dataset(
        features=features,
        labels=labels,
        start_date=train_start_date,
        end_date=train_end_date,
        **dataset_kwargs,
    )
    dataset_kwargs["feature_columns"] = requested_feature_columns or list(
        train.X.columns
    )
    validation = build_modeling_dataset(
        features=features,
        labels=labels,
        start_date=validation_start_date,
        end_date=validation_end_date,
        **dataset_kwargs,
    )

    return TrainValidationDatasets(train=train, validation=validation)


def load_train_validation_datasets(
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
    label_path: str | Path = DEFAULT_LABEL_PATH,
    **kwargs,
) -> TrainValidationDatasets:
    """Load parquet files and construct train/validation modeling datasets."""
    feature_path = Path(feature_path)
    label_path = Path(label_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label table not found: {label_path}")

    features = pd.read_parquet(feature_path)
    labels = pd.read_parquet(label_path)

    return build_train_validation_datasets(features=features, labels=labels, **kwargs)


def _prepare_labels(labels: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    missing_columns = set(KEY_COLUMNS + [target_column]) - set(labels.columns)
    if missing_columns:
        raise ValueError(
            "Label table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = labels.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    if "return_integrity_valid" in result:
        result = result.loc[result["return_integrity_valid"].eq(True)].copy()

    if result["date"].isna().any():
        raise ValueError("Label table contains invalid dates.")

    if result.duplicated(subset=KEY_COLUMNS).any():
        raise ValueError("Label table contains duplicate symbol/date rows.")

    return result


def _filter_dates(
    data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None,
    end_date: str | pd.Timestamp | None,
) -> pd.DataFrame:
    result = data

    if start_date is not None:
        start = pd.Timestamp(start_date).normalize()
        result = result[result["date"] >= start]

    if end_date is not None:
        end = pd.Timestamp(end_date).normalize()
        result = result[result["date"] <= end]

    return result.reset_index(drop=True)


def _infer_feature_columns(
    data: pd.DataFrame,
    *,
    excluded_columns: set[str],
) -> list[str]:
    return [
        column
        for column in data.columns
        if column not in excluded_columns
        and (is_numeric_dtype(data[column]) or is_bool_dtype(data[column]))
    ]


def _infer_metadata_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    feature_column_set = set(feature_columns)
    non_feature_columns = [
        column
        for column in data.columns
        if column not in KEY_COLUMNS and column not in feature_column_set
    ]
    return KEY_COLUMNS + non_feature_columns


def _validate_requested_columns(
    data: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing_columns = set(columns) - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{name} contains missing columns: " + ", ".join(sorted(missing_columns))
        )


def _validate_missing_fraction(max_missing_feature_fraction: float) -> None:
    if not 0 <= max_missing_feature_fraction <= 1:
        raise ValueError("max_missing_feature_fraction must be between 0 and 1.")
