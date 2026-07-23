"""Shared interfaces for quantum machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from market_qml.models.predictions import build_prediction_table

DEFAULT_QML_RANDOM_SEED = 42
DEFAULT_TARGET_COLUMN = "target"
DEFAULT_TRAIN_ROLE = "train"
DEFAULT_VALIDATION_ROLE = "validation"
QML_SAMPLE_REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "split_id",
    "sample_role",
    DEFAULT_TARGET_COLUMN,
]


@dataclass(frozen=True)
class QMLModelConfig:
    """Configuration shared by concrete QML model implementations."""

    model_name: str
    seed: int = DEFAULT_QML_RANDOM_SEED
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QMLDataset:
    """Feature matrix, target vector, and metadata for one QML split role."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class QMLTrainValidation:
    """Train/validation QML datasets built from a sampled QML table."""

    train: QMLDataset
    validation: QMLDataset
    feature_columns: list[str]
    split_id: int


@dataclass(frozen=True)
class QMLModelResult:
    """Fitted QML model and standard validation predictions."""

    model: BaseQMLModel
    predictions: pd.DataFrame
    config: QMLModelConfig


class BaseQMLModel(ABC):
    """Common training and prediction interface for VQC, QSVM, and QCNN models."""

    def __init__(self, config: QMLModelConfig) -> None:
        if not config.model_name:
            raise ValueError("QML model_name must be provided.")
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def seed(self) -> int:
        return self.config.seed

    def train(self, data: QMLTrainValidation) -> QMLModelResult:
        """Fit on train rows and return standard validation predictions."""
        self.fit(data.train)
        y_score = self.predict_scores(data.validation)
        if len(y_score) != len(data.validation.y):
            raise ValueError("QML models must produce one score per validation row.")

        predictions = build_prediction_table(
            metadata=data.validation.metadata,
            y_true=data.validation.y,
            y_score=y_score,
            model_name=self.model_name,
            split_id=data.split_id,
        )
        return QMLModelResult(
            model=self,
            predictions=predictions,
            config=self.config,
        )

    @abstractmethod
    def fit(self, dataset: QMLDataset) -> BaseQMLModel:
        """Fit the model on one QML training dataset."""

    @abstractmethod
    def predict_scores(self, dataset: QMLDataset) -> Sequence[float]:
        """Predict continuous validation scores for ranking/backtests."""


def build_qml_train_validation(
    qml_sample: pd.DataFrame,
    *,
    split_id: int = 0,
    feature_columns: list[str] | None = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    train_role: str = DEFAULT_TRAIN_ROLE,
    validation_role: str = DEFAULT_VALIDATION_ROLE,
) -> QMLTrainValidation:
    """Build the expected QML model input format from sampled QML rows."""
    _validate_sample_columns(qml_sample, target_column=target_column)

    data = qml_sample[qml_sample["split_id"] == split_id].copy()
    if data.empty:
        raise ValueError(f"QML sample does not contain split_id={split_id}.")

    if feature_columns is None:
        feature_columns = _infer_qml_feature_columns(data)
    else:
        _validate_requested_columns(data, feature_columns, "feature_columns")

    if not feature_columns:
        raise ValueError("QML sample does not contain PCA component columns.")

    train = _dataset_for_role(
        data,
        role=train_role,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    validation = _dataset_for_role(
        data,
        role=validation_role,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    return QMLTrainValidation(
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        split_id=split_id,
    )


def _dataset_for_role(
    data: pd.DataFrame,
    *,
    role: str,
    feature_columns: list[str],
    target_column: str,
) -> QMLDataset:
    role_data = data[data["sample_role"] == role].copy()
    if role_data.empty:
        raise ValueError(f"QML sample does not contain {role} rows.")

    X = role_data[feature_columns].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(f"QML {role} features contain missing or non-numeric values.")

    y = pd.to_numeric(role_data[target_column], errors="coerce")
    if y.isna().any():
        raise ValueError(f"QML {role} targets contain missing or non-numeric values.")

    metadata_columns = [
        column
        for column in role_data.columns
        if column not in set(feature_columns + [target_column])
    ]
    metadata = role_data[metadata_columns].copy()

    return QMLDataset(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
    )


def _infer_qml_feature_columns(data: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in data.columns
        if column.startswith("pca_") or "_pca_" in column
    )


def _validate_sample_columns(
    qml_sample: pd.DataFrame,
    *,
    target_column: str,
) -> None:
    required_columns = set(QML_SAMPLE_REQUIRED_COLUMNS)
    required_columns.remove(DEFAULT_TARGET_COLUMN)
    required_columns.add(target_column)
    missing_columns = required_columns - set(qml_sample.columns)
    if missing_columns:
        raise ValueError(
            "QML sample is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


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
