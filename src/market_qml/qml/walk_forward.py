"""Reusable QML preparation for walk-forward model execution."""

from __future__ import annotations

import pandas as pd

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.pca import fit_pca


def build_qml_split_sample(*, preprocessed, split_id: int, n_components: int):
    pca = fit_pca(preprocessed.train.X, n_components=n_components)
    train_rows = pca_rows(
        X=preprocessed.train.X,
        y=preprocessed.train.y,
        metadata=preprocessed.train.metadata,
        pca=pca,
        split_id=split_id,
        sample_role="train",
        n_components=n_components,
    )
    validation_rows = pca_rows(
        X=preprocessed.validation.X,
        y=preprocessed.validation.y,
        metadata=preprocessed.validation.metadata,
        pca=pca,
        split_id=split_id,
        sample_role="validation",
        n_components=n_components,
    )
    sample = pd.concat([train_rows, validation_rows], ignore_index=True)
    return build_qml_train_validation(sample, split_id=split_id), pca


def pca_rows(
    *, X, y, metadata, pca, split_id: int, sample_role: str, n_components: int
) -> pd.DataFrame:
    columns = [f"pca_{index:02d}" for index in range(n_components)]
    components = pd.DataFrame(pca.transform(X), columns=columns, index=X.index)
    result = metadata.copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["split_id"] = split_id
    result["sample_role"] = sample_role
    result["target"] = pd.to_numeric(y, errors="coerce").to_numpy()
    result = pd.concat([result, components.reset_index(drop=True)], axis=1)
    if result["date"].isna().any():
        raise ValueError("QML walk-forward PCA rows contain invalid dates.")
    if result["target"].isna().any():
        raise ValueError("QML walk-forward PCA rows contain invalid targets.")
    return result
