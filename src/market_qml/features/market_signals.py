"""Causal price-path, residual-momentum, and liquidity signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"symbol", "date", "close", "high", "low", "volume", "return_1d"}


def add_market_signal_features(
    features: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    windows: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Add trailing-only market signals to an accumulated feature table."""
    missing = REQUIRED_COLUMNS - set(features)
    if missing:
        raise ValueError("Market signals are missing: " + ", ".join(sorted(missing)))
    if not windows or any(window <= 2 for window in windows):
        raise ValueError("Market-signal windows must be greater than two.")

    result = features.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in ["close", "high", "low", "volume", "return_1d"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("Market signals contain invalid dates.")
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

    benchmark = result.loc[
        result["symbol"].eq(benchmark_symbol.upper()), ["date", "return_1d"]
    ].rename(columns={"return_1d": "_market_return_1d"})
    if benchmark.empty or benchmark["date"].duplicated().any():
        raise ValueError(f"A unique {benchmark_symbol.upper()} history is required.")
    result = result.merge(benchmark, on="date", how="left", validate="many_to_one")
    grouped = result.groupby("symbol", sort=False)

    previous_close = grouped["close"].shift(1)
    result["price_range_1d"] = (result["high"] - result["low"]) / previous_close
    result["amihud_illiquidity_1d"] = result["return_1d"].abs() / (
        result["close"] * result["volume"]
    ).replace(0, np.nan)
    volume_change = grouped["volume"].pct_change(fill_method=None)

    for window in windows:
        rolling = grouped["return_1d"]
        result[f"downside_vol_{window}d"] = rolling.transform(
            lambda values: (
                values.where(values < 0)
                .rolling(window, min_periods=window // 2)
                .std(ddof=0)
            )
        )
        result[f"positive_day_share_{window}d"] = rolling.transform(
            lambda values: values.gt(0).rolling(window, min_periods=window).mean()
        )
        result[f"zero_return_share_{window}d"] = rolling.transform(
            lambda values: values.eq(0).rolling(window, min_periods=window).mean()
        )
        result[f"amihud_illiquidity_{window}d"] = grouped[
            "amihud_illiquidity_1d"
        ].transform(lambda values: values.rolling(window, min_periods=window).mean())
        result[f"range_volatility_{window}d"] = grouped["price_range_1d"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        result[f"drawdown_{window}d"] = (
            result["close"]
            / grouped["close"].transform(
                lambda values: values.rolling(window, min_periods=window).max()
            )
            - 1
        )
        result[f"volume_return_confirmation_{window}d"] = _rolling_pair_correlation(
            result, volume_change, window
        )

        covariance = result.groupby("symbol", sort=False, group_keys=False)[
            ["return_1d", "_market_return_1d"]
        ].apply(
            lambda frame: (
                frame["return_1d"]
                .rolling(window, min_periods=window)
                .cov(frame["_market_return_1d"], ddof=0)
            )
        )
        market_variance = grouped["_market_return_1d"].transform(
            lambda values: values.rolling(window, min_periods=window).var(ddof=0)
        )
        beta = covariance / market_variance.replace(0, np.nan)
        residual = result["return_1d"] - beta * result["_market_return_1d"]
        result[f"residual_momentum_{window}d"] = residual.groupby(
            result["symbol"]
        ).transform(lambda values: values.rolling(window, min_periods=window).sum())

    result["reversal_1d"] = -result["return_1d"]
    if "return_5d" in result:
        result["reversal_5d"] = -pd.to_numeric(result["return_5d"], errors="coerce")
    result = result.drop(columns="_market_return_1d")
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def _rolling_pair_correlation(
    frame: pd.DataFrame, right: pd.Series, window: int
) -> pd.Series:
    temporary = frame[["symbol", "return_1d"]].copy()
    temporary["_right"] = right
    return temporary.groupby("symbol", sort=False, group_keys=False)[
        ["return_1d", "_right"]
    ].apply(
        lambda group: (
            group["return_1d"].rolling(window, min_periods=window).corr(group["_right"])
        )
    )
