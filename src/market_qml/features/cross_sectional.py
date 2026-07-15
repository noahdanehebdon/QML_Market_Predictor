"""Leakage-safe cross-sectional feature engineering."""

from __future__ import annotations

import pandas as pd


DEFAULT_RANK_FEATURES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "realized_vol_20d",
    "volume_shock_20d",
    "relative_momentum_20d_vs_spy",
    "relative_momentum_60d_vs_spy",
)


def add_cross_sectional_features(
    features: pd.DataFrame,
    *,
    rank_features: tuple[str, ...] = DEFAULT_RANK_FEATURES,
) -> pd.DataFrame:
    """Add same-date percentile ranks and missingness indicators.

    Percentiles are computed independently on each date, using only the
    contemporaneous investable cross-section. Missingness flags let models
    distinguish an imputed value from an observed value downstream.
    """
    if "date" not in features.columns:
        raise ValueError("Cross-sectional features require a date column.")

    result = features.copy()
    for column in rank_features:
        if column not in result.columns:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_missing"] = numeric.isna().astype("int8")
        result[f"{column}_xs_rank"] = numeric.groupby(result["date"]).rank(
            method="average",
            pct=True,
        )
    return result
