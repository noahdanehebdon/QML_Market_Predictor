"""Train-only preprocessing for modeling datasets."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.models.dataset import ModelingDataset, TrainValidationDatasets

DEFAULT_PREPROCESSOR_PATH = Path("artifacts/preprocessing/preprocessor.pkl")


@dataclass(frozen=True)
class FittedPreprocessor:
    """Reusable preprocessing statistics fitted on training data only."""

    feature_columns: list[str]
    fill_values: pd.Series
    means: pd.Series
    scales: pd.Series


@dataclass(frozen=True)
class PreprocessedDataset:
    """Preprocessed feature matrix with original target and metadata."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class PreprocessedTrainValidation:
    """Preprocessed train and validation datasets plus fitted artifact."""

    train: PreprocessedDataset
    validation: PreprocessedDataset
    preprocessor: FittedPreprocessor


def fit_preprocessor(train_X: pd.DataFrame) -> FittedPreprocessor:
    """Fit missing-value and standardization statistics on training features."""
    if train_X.empty:
        raise ValueError("Cannot fit preprocessor on an empty training matrix.")

    numeric = _coerce_numeric_frame(train_X)
    feature_columns = list(numeric.columns)

    if not feature_columns:
        raise ValueError("Training matrix does not contain feature columns.")

    fill_values = numeric.median(axis=0, skipna=True).fillna(0.0)
    imputed = numeric.fillna(fill_values)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    _require_finite(imputed, context="imputed training features")
    _require_finite(means.to_frame().T, context="training feature means")
    _require_finite(scales.to_frame().T, context="training feature scales")

    return FittedPreprocessor(
        feature_columns=feature_columns,
        fill_values=fill_values,
        means=means,
        scales=scales,
    )


def transform_features(
    X: pd.DataFrame,
    preprocessor: FittedPreprocessor,
) -> pd.DataFrame:
    """Apply a train-fitted preprocessor to a feature matrix."""
    missing_columns = set(preprocessor.feature_columns) - set(X.columns)
    if missing_columns:
        raise ValueError(
            "Feature matrix is missing fitted columns: "
            + ", ".join(sorted(missing_columns))
        )

    numeric = _coerce_numeric_frame(X[preprocessor.feature_columns])
    imputed = numeric.fillna(preprocessor.fill_values)
    transformed = (imputed - preprocessor.means) / preprocessor.scales
    transformed = transformed[preprocessor.feature_columns]
    _require_finite(transformed, context="transformed features")
    return transformed


def preprocess_dataset(
    dataset: ModelingDataset,
    preprocessor: FittedPreprocessor,
) -> PreprocessedDataset:
    """Transform one modeling dataset with a fitted preprocessor."""
    return PreprocessedDataset(
        X=transform_features(dataset.X, preprocessor),
        y=dataset.y.copy(),
        metadata=dataset.metadata.copy(),
    )


def fit_transform_train_validation(
    datasets: TrainValidationDatasets,
) -> PreprocessedTrainValidation:
    """Fit preprocessing on train only and apply it to train and validation."""
    preprocessor = fit_preprocessor(datasets.train.X)
    return PreprocessedTrainValidation(
        train=preprocess_dataset(datasets.train, preprocessor),
        validation=preprocess_dataset(datasets.validation, preprocessor),
        preprocessor=preprocessor,
    )


def save_preprocessor(
    preprocessor: FittedPreprocessor,
    output_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
) -> None:
    """Save a fitted preprocessor artifact for reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(preprocessor, f)


def load_preprocessor(
    path: str | Path = DEFAULT_PREPROCESSOR_PATH,
) -> FittedPreprocessor:
    """Load a fitted preprocessor artifact."""
    path = Path(path)
    with path.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, FittedPreprocessor):
        raise TypeError(f"Unexpected preprocessing artifact type: {type(artifact)!r}")

    return artifact


def _coerce_numeric_frame(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    return numeric.astype("float64").replace([np.inf, -np.inf], np.nan)


def _require_finite(frame: pd.DataFrame, *, context: str) -> None:
    finite = np.isfinite(frame.to_numpy(dtype="float64"))
    if finite.all():
        return
    invalid_columns = frame.columns[~finite.all(axis=0)].astype(str).tolist()
    raise ValueError(
        f"Preprocessing produced non-finite {context} in columns: "
        + ", ".join(invalid_columns)
    )
