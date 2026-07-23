"""Leakage-safe daily market regime definitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"symbol", "date", "return_1d", "treasury_10y", "treasury_2y"}
REGIME_COLUMNS = [
    "date",
    "spy_realized_volatility",
    "spy_volatility_threshold",
    "volatility_regime",
    "treasury_yield_level",
    "treasury_yield_change",
    "rate_regime",
    "yield_spread_10y_2y",
    "yield_spread_change",
    "yield_curve_regime",
    "yield_curve_trend",
]


def build_market_regimes(
    features: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    volatility_window: int = 20,
    rate_window: int = 20,
    annualization_factor: int = 252,
    minimum_threshold_history: int | None = None,
    curve_flat_tolerance: float = 0.0,
) -> pd.DataFrame:
    """Create one leakage-safe market-regime row per trading date.

    Volatility is high or low relative to the expanding median of *prior* valid
    SPY rolling-volatility observations. Rate direction uses the trailing change
    in the mean 2Y/10Y Treasury yield. Curve shape uses the contemporaneously
    available 10Y-minus-2Y spread; curve direction uses its trailing change.
    """
    _validate_inputs(
        features,
        volatility_window=volatility_window,
        rate_window=rate_window,
        annualization_factor=annualization_factor,
        curve_flat_tolerance=curve_flat_tolerance,
    )
    threshold_history = minimum_threshold_history or volatility_window
    if threshold_history <= 0:
        raise ValueError("minimum_threshold_history must be positive")

    normalized = features.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="coerce"
    ).dt.normalize()
    if normalized["date"].isna().any():
        raise ValueError("Regime features contain invalid dates")

    benchmark = normalized.loc[
        normalized["symbol"].astype(str) == benchmark_symbol,
        ["date", "return_1d"],
    ].sort_values("date")
    if benchmark.empty:
        raise ValueError(f"Benchmark symbol not found: {benchmark_symbol}")
    if benchmark["date"].duplicated().any():
        raise ValueError(f"Benchmark has duplicate dates: {benchmark_symbol}")

    benchmark_returns = pd.to_numeric(benchmark["return_1d"], errors="coerce")
    volatility = benchmark_returns.rolling(
        volatility_window, min_periods=volatility_window
    ).std(ddof=1) * np.sqrt(annualization_factor)
    threshold = volatility.expanding(min_periods=threshold_history).median().shift(1)

    daily = _daily_rates(normalized)
    result = benchmark[["date"]].merge(
        daily, on="date", how="left", validate="one_to_one"
    )
    result["spy_realized_volatility"] = volatility.to_numpy()
    result["spy_volatility_threshold"] = threshold.to_numpy()
    result["volatility_regime"] = _binary_regime(
        result["spy_realized_volatility"] - result["spy_volatility_threshold"],
        positive="high_volatility",
        negative="low_volatility",
        zero="normal_volatility",
    )

    result["treasury_yield_level"] = result[["treasury_10y", "treasury_2y"]].mean(
        axis=1
    )
    result["treasury_yield_change"] = result["treasury_yield_level"].diff(rate_window)
    result["rate_regime"] = _binary_regime(
        result["treasury_yield_change"],
        positive="rising_rates",
        negative="falling_rates",
        zero="flat_rates",
    )

    result["yield_spread_10y_2y"] = result["treasury_10y"] - result["treasury_2y"]
    spread = result["yield_spread_10y_2y"]
    result["yield_curve_regime"] = pd.Series(
        np.select(
            [spread > curve_flat_tolerance, spread < -curve_flat_tolerance],
            ["normal_curve", "inverted_curve"],
            default="flat_curve",
        ),
        index=result.index,
        dtype="string",
    ).mask(spread.isna())
    result["yield_spread_change"] = spread.diff(rate_window)
    result["yield_curve_trend"] = _binary_regime(
        result["yield_spread_change"],
        positive="steepening_curve",
        negative="flattening_curve",
        zero="unchanged_curve",
    )
    return result[REGIME_COLUMNS]


def save_market_regimes(regimes: pd.DataFrame, output_path: str | Path) -> None:
    """Save date-keyed regime labels to parquet."""
    missing = set(REGIME_COLUMNS) - set(regimes.columns)
    if missing:
        raise ValueError(
            "Regime table is missing columns: " + ", ".join(sorted(missing))
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    regimes[REGIME_COLUMNS].to_parquet(output, index=False)


def _daily_rates(features: pd.DataFrame) -> pd.DataFrame:
    rates = features[["date", "treasury_10y", "treasury_2y"]].copy()
    rates[["treasury_10y", "treasury_2y"]] = rates[
        ["treasury_10y", "treasury_2y"]
    ].apply(pd.to_numeric, errors="coerce")
    conflicts = rates.groupby("date")[["treasury_10y", "treasury_2y"]].nunique(
        dropna=False
    )
    if conflicts.gt(1).any(axis=None):
        raise ValueError(
            "Treasury yields must be identical across symbols on each date"
        )
    return rates.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _binary_regime(
    values: pd.Series, *, positive: str, negative: str, zero: str
) -> pd.Series:
    labels = pd.Series(
        np.select([values > 0, values < 0], [positive, negative], default=zero),
        index=values.index,
        dtype="string",
    )
    return labels.mask(values.isna())


def _validate_inputs(
    features,
    *,
    volatility_window,
    rate_window,
    annualization_factor,
    curve_flat_tolerance,
):
    missing = REQUIRED_COLUMNS - set(features.columns)
    if missing:
        raise ValueError(
            "Regime features are missing columns: " + ", ".join(sorted(missing))
        )
    if features.empty:
        raise ValueError("Regime feature table is empty")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be greater than one")
    if rate_window <= 0:
        raise ValueError("rate_window must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if curve_flat_tolerance < 0:
        raise ValueError("curve_flat_tolerance must be non-negative")
