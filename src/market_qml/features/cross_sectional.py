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
    sector_column: str = "sector",
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
        date_group = numeric.groupby(result["date"])
        median = date_group.transform("median")
        mad = (numeric - median).abs().groupby(result["date"]).transform("median")
        scale = (1.4826 * mad).where(mad > 0)
        result[f"{column}_xs_robust_z"] = ((numeric - median) / scale).clip(-5, 5)
        if sector_column in result.columns:
            sectors = result[sector_column].astype("string")
            valid_sector = sectors.notna()
            sector_rank = pd.Series(pd.NA, index=result.index, dtype="Float64")
            sector_rank.loc[valid_sector] = (
                numeric.loc[valid_sector]
                .groupby([result.loc[valid_sector, "date"], sectors.loc[valid_sector]])
                .rank(method="average", pct=True)
            )
            result[f"{column}_sector_rank"] = sector_rank
    return result
