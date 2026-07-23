"""Angle encoding utilities for QML feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, pi

import pandas as pd

from market_qml.qml.interface import QMLDataset

ANGLE_MIN = -pi
ANGLE_MAX = pi
DEFAULT_ANGLE_PREFIX = "theta"
DEFAULT_ANGLE_GATE = "ry"


@dataclass(frozen=True)
class AngleEncodingConfig:
    """Configuration for framework-neutral angle encoding.

    PCA components are unbounded real values. This encoder uses ``2 * atan(x)``
    so every component maps smoothly into the valid rotation range ``[-pi, pi]``
    without fitting scaling parameters that could leak validation information.
    """

    n_qubits: int = 8
    angle_prefix: str = DEFAULT_ANGLE_PREFIX
    gate: str = DEFAULT_ANGLE_GATE


@dataclass(frozen=True)
class AngleEncodingResult:
    """Encoded angle matrix and reusable per-qubit operation metadata."""

    angles: pd.DataFrame
    operations: pd.DataFrame
    feature_columns: list[str]
    config: AngleEncodingConfig


def angle_encode_features(
    features: pd.DataFrame,
    *,
    config: AngleEncodingConfig | None = None,
    feature_columns: list[str] | None = None,
) -> AngleEncodingResult:
    """Scale QML features to angles and describe one rotation per qubit."""
    config = config or AngleEncodingConfig()
    _validate_config(config)

    if feature_columns is None:
        feature_columns = _infer_angle_feature_columns(features)
    else:
        _validate_requested_columns(features, feature_columns, "feature_columns")

    if len(feature_columns) != config.n_qubits:
        raise ValueError(
            f"Angle encoding requires exactly {config.n_qubits} feature columns."
        )

    numeric = features[feature_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(
            "Angle encoding features contain missing or non-numeric values."
        )

    angle_columns = _angle_columns(config)
    angles = numeric.map(_scale_value_to_angle)
    angles.columns = angle_columns
    operations = angle_encoding_operations(
        feature_columns=feature_columns,
        angle_columns=angle_columns,
        config=config,
    )

    return AngleEncodingResult(
        angles=angles.reset_index(drop=True),
        operations=operations,
        feature_columns=feature_columns,
        config=config,
    )


def angle_encode_dataset(
    dataset: QMLDataset,
    *,
    config: AngleEncodingConfig | None = None,
    feature_columns: list[str] | None = None,
) -> QMLDataset:
    """Return a QML dataset whose feature matrix contains encoded angles."""
    result = angle_encode_features(
        dataset.X,
        config=config,
        feature_columns=feature_columns,
    )
    return QMLDataset(
        X=result.angles,
        y=dataset.y.copy().reset_index(drop=True),
        metadata=dataset.metadata.copy().reset_index(drop=True),
    )


def angle_encoding_operations(
    *,
    feature_columns: list[str],
    angle_columns: list[str],
    config: AngleEncodingConfig,
) -> pd.DataFrame:
    """Build framework-neutral rotation metadata for VQC and QCNN circuits."""
    if len(feature_columns) != len(angle_columns):
        raise ValueError("feature_columns and angle_columns must have equal length.")

    return pd.DataFrame(
        {
            "qubit": list(range(len(feature_columns))),
            "gate": [config.gate] * len(feature_columns),
            "feature_column": feature_columns,
            "angle_column": angle_columns,
        }
    )


def scale_value_to_angle(value: float) -> float:
    """Map one unbounded PCA value into the rotation range ``[-pi, pi]``."""
    return _scale_value_to_angle(float(value))


def _scale_value_to_angle(value: float) -> float:
    return 2.0 * atan(value)


def _infer_angle_feature_columns(features: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in features.columns
        if column.startswith("pca_") or "_pca_" in column
    )


def _angle_columns(config: AngleEncodingConfig) -> list[str]:
    return [f"{config.angle_prefix}_{index:02d}" for index in range(config.n_qubits)]


def _validate_config(config: AngleEncodingConfig) -> None:
    if config.n_qubits <= 0:
        raise ValueError("n_qubits must be a positive integer.")
    if not config.angle_prefix:
        raise ValueError("angle_prefix must be provided.")
    if not config.gate:
        raise ValueError("gate must be provided.")


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
