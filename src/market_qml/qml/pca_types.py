"""Data contracts for leakage-safe QML PCA compression."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA

from market_qml.models.preprocessing import FittedPreprocessor


@dataclass(frozen=True)
class PCAArtifact:
    """Train-fitted preprocessing and PCA state for one split."""

    split_id: int
    target_column: str
    feature_columns: list[str]
    preprocessor: FittedPreprocessor
    pca: PCA


@dataclass(frozen=True)
class GroupedPCAArtifact:
    """Train-fitted preprocessing and grouped PCA state for one split."""

    split_id: int
    target_column: str
    feature_columns: list[str]
    group_columns: dict[str, list[str]]
    component_columns: list[str]
    preprocessor: FittedPreprocessor
    pcas: dict[str, PCA]


@dataclass(frozen=True)
class QMLPCACompressionResult:
    """Compressed QML feature rows, diagnostics, and fitted PCA artifacts."""

    features: pd.DataFrame
    diagnostics: pd.DataFrame
    artifacts: dict[int, PCAArtifact | GroupedPCAArtifact]
