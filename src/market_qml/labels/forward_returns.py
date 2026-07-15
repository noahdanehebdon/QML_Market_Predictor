"""Forward return label engineering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LABEL_HORIZON = 5
DEFAULT_BENCHMARK_SYMBOL = "SPY"
REQUIRED_PRICE_COLUMNS = {"symbol", "date", "close"}


def build_forward_return_labels(
    prices: pd.DataFrame,
    *,
    horizon: int = DEFAULT_LABEL_HORIZON,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    drop_missing: bool = True,
    neutral_threshold: float = 0.0,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Create forward excess return labels from close prices.

    Forward returns are computed independently by symbol as:
    close[t + horizon] / close[t] - 1.
    """
    missing_columns = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing_columns:
        raise ValueError(
            "Price table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if horizon <= 0:
        raise ValueError(f"Label horizon must be positive: {horizon}")
    if neutral_threshold < 0:
        raise ValueError("Neutral threshold cannot be negative.")
    if volatility_window <= 1:
        raise ValueError("Volatility window must be greater than one.")

    benchmark_symbol = benchmark_symbol.strip().upper()
    if not benchmark_symbol:
        raise ValueError("Benchmark symbol cannot be empty.")

    labels = prices.copy()
    labels["symbol"] = labels["symbol"].astype(str).str.upper()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.normalize()
    labels["close"] = pd.to_numeric(labels["close"], errors="coerce")

    if labels["date"].isna().any():
        raise ValueError("Price table contains invalid dates.")

    labels = labels.dropna(subset=["symbol", "close"])
    labels = labels.sort_values(["symbol", "date"]).reset_index(drop=True)

    benchmark_rows = labels[labels["symbol"] == benchmark_symbol]
    if benchmark_rows.empty:
        raise ValueError(f"Benchmark symbol not found in price table: {benchmark_symbol}")

    forward_return_column = f"forward_return_{horizon}d"
    benchmark_return_column = f"{benchmark_symbol.lower()}_forward_return_{horizon}d"
    excess_return_column = f"forward_excess_return_{horizon}d"
    binary_label_column = f"outperform_{benchmark_symbol.lower()}_{horizon}d"
    neutral_label_column = f"outperform_{benchmark_symbol.lower()}_{horizon}d_neutral"
    normalized_return_column = f"vol_normalized_excess_return_{horizon}d"

    labels["_daily_return"] = labels.groupby("symbol", sort=False)["close"].pct_change(
        fill_method=None
    )
    benchmark_daily = labels.loc[
        labels["symbol"] == benchmark_symbol, ["date", "_daily_return"]
    ].rename(columns={"_daily_return": "_benchmark_daily_return"})
    labels = labels.merge(benchmark_daily, on="date", how="left")
    labels["_daily_excess_return"] = (
        labels["_daily_return"] - labels["_benchmark_daily_return"]
    )
    labels["_excess_volatility"] = labels.groupby("symbol", sort=False)[
        "_daily_excess_return"
    ].transform(
        lambda values: values.rolling(
            volatility_window,
            min_periods=max(2, volatility_window // 2),
        ).std()
    )

    labels[forward_return_column] = labels.groupby("symbol", sort=False)[
        "close"
    ].transform(lambda close: close.shift(-horizon) / close - 1)

    benchmark_returns = labels.loc[
        labels["symbol"] == benchmark_symbol,
        ["date", forward_return_column],
    ].rename(columns={forward_return_column: benchmark_return_column})

    result = labels[["symbol", "date", forward_return_column]].merge(
        benchmark_returns,
        on="date",
        how="left",
    )
    result["label_horizon_days"] = horizon
    result[excess_return_column] = (
        result[forward_return_column] - result[benchmark_return_column]
    )
    volatility = labels[["symbol", "date", "_excess_volatility"]]
    result = result.merge(volatility, on=["symbol", "date"], how="left")
    denominator = result["_excess_volatility"] * np.sqrt(horizon)
    result[normalized_return_column] = result[excess_return_column] / denominator
    result[normalized_return_column] = result[normalized_return_column].replace(
        [np.inf, -np.inf], np.nan
    )

    result[binary_label_column] = pd.NA
    valid_excess = result[excess_return_column].notna()
    result.loc[valid_excess, binary_label_column] = (
        result.loc[valid_excess, excess_return_column] > 0
    ).astype("int64")
    result[binary_label_column] = result[binary_label_column].astype("Int64")
    result[neutral_label_column] = pd.NA
    decisive = valid_excess & (
        result[excess_return_column].abs() > neutral_threshold
    )
    result.loc[decisive, neutral_label_column] = (
        result.loc[decisive, excess_return_column] > neutral_threshold
    ).astype("int64")
    result[neutral_label_column] = result[neutral_label_column].astype("Int64")

    ordered_columns = [
        "symbol",
        "date",
        "label_horizon_days",
        forward_return_column,
        benchmark_return_column,
        excess_return_column,
        normalized_return_column,
        binary_label_column,
        neutral_label_column,
    ]
    result = result[ordered_columns].sort_values(["symbol", "date"]).reset_index(drop=True)

    if drop_missing:
        result = result.dropna(
            subset=[
                forward_return_column,
                benchmark_return_column,
                excess_return_column,
                binary_label_column,
            ]
        ).reset_index(drop=True)

    return result


def build_forward_return_label_table(
    price_path: str | Path,
    output_path: str | Path,
    *,
    horizon: int = DEFAULT_LABEL_HORIZON,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    drop_missing: bool = True,
    neutral_threshold: float = 0.0,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Load prices, build forward return labels, save, and return them."""
    price_path = Path(price_path)
    output_path = Path(output_path)

    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    prices = pd.read_parquet(price_path)
    labels = build_forward_return_labels(
        prices=prices,
        horizon=horizon,
        benchmark_symbol=benchmark_symbol,
        drop_missing=drop_missing,
        neutral_threshold=neutral_threshold,
        volatility_window=volatility_window,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)

    return labels
