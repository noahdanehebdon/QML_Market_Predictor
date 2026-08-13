"""Economically interpretable interactions between common regimes and exposures."""

from __future__ import annotations

import pandas as pd

INTERACTIONS = {
    "rate_beta_interaction_20d": ("rolling_beta_20d_vs_spy", "treasury_10y_change_20d"),
    "curve_leverage_interaction_20d": (
        "liability_ratio",
        "yield_spread_10y_2y_change_20d",
    ),
    "rate_debt_interaction_20d": ("debt_ratio", "treasury_10y_change_20d"),
    "inflation_margin_interaction_63d": ("operating_margin", "cpi_inflation_63d"),
    "stress_liquidity_interaction_20d": (
        "amihud_illiquidity_20d",
        "realized_vol_20d",
    ),
    "stress_residual_momentum_interaction_20d": (
        "residual_momentum_20d",
        "realized_vol_20d",
    ),
}


def add_conditional_features(features: pd.DataFrame) -> pd.DataFrame:
    """Multiply point-in-time exposures by contemporaneously available regimes."""
    result = features.copy()
    for output, (exposure, regime) in INTERACTIONS.items():
        if exposure not in result or regime not in result:
            continue
        left = pd.to_numeric(result[exposure], errors="coerce")
        right = pd.to_numeric(result[regime], errors="coerce")
        result[output] = left * right
    return result
