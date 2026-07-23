"""Training-only feature and hyperparameter selection for boosted regression."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from market_qml.models.predictions import build_prediction_table
from market_qml.models.preprocessing import PreprocessedTrainValidation
from market_qml.utils.statistics import safe_correlation

MODEL_NAME = "tuned_gradient_boosting_regressor"
DEFAULT_TARGET_COLUMN = "forward_excess_return_5d"


@dataclass(frozen=True)
class TunedGradientBoostingResult:
    model: HistGradientBoostingRegressor
    predictions: pd.DataFrame
    parameters: dict[str, object]
    selection_diagnostics: pd.DataFrame


def train_tuned_gradient_boosting_regressor(
    data: PreprocessedTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int = 0,
    feature_counts: tuple[int | None, ...] = (20, 50, None),
    learning_rates: tuple[float, ...] = (0.03, 0.08),
    max_leaf_nodes_values: tuple[int, ...] = (15, 31),
    l2_values: tuple[float, ...] = (0.0, 0.1),
    inner_validation_fraction: float = 0.2,
    min_samples_leaf: int = 20,
    max_iter: int = 300,
    random_state: int = 42,
) -> TunedGradientBoostingResult:
    """Select configuration on the chronological tail of outer training only."""
    if not 0 < inner_validation_fraction < 0.5:
        raise ValueError("inner_validation_fraction must be between 0 and 0.5.")

    dates = pd.to_datetime(data.train.metadata["date"], errors="coerce")
    unique_dates = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    cutoff_index = max(1, int(len(unique_dates) * (1 - inner_validation_fraction)))
    if cutoff_index >= len(unique_dates):
        raise ValueError("Training data has too few dates for inner validation.")
    cutoff = unique_dates[cutoff_index]
    inner_train_mask = dates < cutoff
    inner_validation_mask = dates >= cutoff

    X_inner_train = data.train.X.loc[inner_train_mask].reset_index(drop=True)
    y_inner_train = pd.to_numeric(
        data.train.y.loc[inner_train_mask], errors="coerce"
    ).reset_index(drop=True)
    X_inner_validation = data.train.X.loc[inner_validation_mask].reset_index(drop=True)
    y_inner_validation = pd.to_numeric(
        data.train.y.loc[inner_validation_mask], errors="coerce"
    ).reset_index(drop=True)
    if y_inner_train.isna().any() or y_inner_validation.isna().any():
        raise ValueError("Training targets contain missing or non-numeric values.")

    ranked_features = _rank_features(X_inner_train, y_inner_train)
    rows: list[dict[str, object]] = []
    configurations = product(
        feature_counts, learning_rates, max_leaf_nodes_values, l2_values
    )
    for config_id, (count, rate, leaves, l2) in enumerate(configurations):
        selected = ranked_features[: _resolved_count(count, len(ranked_features))]
        model = HistGradientBoostingRegressor(
            learning_rate=rate,
            max_iter=max_iter,
            max_leaf_nodes=leaves,
            l2_regularization=l2,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(X_inner_train[selected], y_inner_train)
        inner_scores = pd.Series(model.predict(X_inner_validation[selected]))
        score = (
            inner_scores.corr(y_inner_validation, method="spearman")
            if inner_scores.nunique() > 1 and y_inner_validation.nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "config_id": config_id,
                "feature_count": len(selected),
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
                "inner_rank_ic": float(score) if pd.notna(score) else -1.0,
                "inner_train_rows": len(X_inner_train),
                "inner_validation_rows": len(X_inner_validation),
                "inner_validation_start": cutoff,
            }
        )

    diagnostics = (
        pd.DataFrame(rows)
        .sort_values(
            ["inner_rank_ic", "feature_count", "config_id"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    diagnostics.insert(0, "rank", np.arange(1, len(diagnostics) + 1))
    diagnostics["split_id"] = split_id
    best = diagnostics.iloc[0]

    # Recompute supervised feature scores using all outer-training rows only.
    selected_features = _rank_features(
        data.train.X,
        pd.to_numeric(data.train.y, errors="coerce"),
    )[: int(best["feature_count"])]
    final_model = HistGradientBoostingRegressor(
        learning_rate=float(best["learning_rate"]),
        max_iter=max_iter,
        max_leaf_nodes=int(best["max_leaf_nodes"]),
        l2_regularization=float(best["l2_regularization"]),
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    final_model.fit(data.train.X[selected_features], data.train.y)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=final_model.predict(data.validation.X[selected_features]),
        model_name=model_name,
        split_id=split_id,
    )
    parameters = {
        "model": model_name,
        "selected_features": selected_features,
        "feature_count": len(selected_features),
        "learning_rate": float(best["learning_rate"]),
        "max_leaf_nodes": int(best["max_leaf_nodes"]),
        "l2_regularization": float(best["l2_regularization"]),
        "inner_rank_ic": float(best["inner_rank_ic"]),
    }
    return TunedGradientBoostingResult(
        model=final_model,
        predictions=predictions,
        parameters=parameters,
        selection_diagnostics=diagnostics,
    )


def _rank_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    # Absolute correlation is stable for perfectly linear features, where an
    # F-statistic can overflow because the residual variance approaches zero.
    scores = pd.Series(
        {
            column: abs(correlation)
            if np.isfinite(correlation := safe_correlation(X[column], y))
            else -np.inf
            for column in X
        }
    )
    return [str(column) for column in scores.sort_values(ascending=False).index]


def _resolved_count(count: int | None, available: int) -> int:
    if count is None:
        return available
    return max(1, min(count, available))
