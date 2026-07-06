"""Price return feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RETURN_WINDOWS = [1, 5, 10, 20, 60]
REQUIRED_PRICE_COLUMNS = {"symbol", "date", "close"}


def add_return_features(
    prices: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add backward-looking return features to a price table.

    Returns are computed independently by symbol as:
    close[t] / close[t - window] - 1.
    """
    windows = windows or RETURN_WINDOWS
    missing_columns = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing_columns:
        raise ValueError(
            "Price table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not windows:
        raise ValueError("At least one return window is required.")

    invalid_windows = [window for window in windows if window <= 0]
    if invalid_windows:
        raise ValueError(
            "Return windows must be positive integers: "
            + ", ".join(str(window) for window in invalid_windows)
        )

    features = prices.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["close"] = pd.to_numeric(features["close"], errors="coerce")

    if features["date"].isna().any():
        raise ValueError("Price table contains invalid dates.")

    features = features.sort_values(["symbol", "date"]).reset_index(drop=True)
    close_by_symbol = features.groupby("symbol", sort=False)["close"]

    for window in windows:
        features[f"return_{window}d"] = close_by_symbol.pct_change(periods=window)

    return features


def build_price_return_features(
    price_path: str | Path,
    output_path: str | Path,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Load prices, add return features, save the feature table, and return it."""
    price_path = Path(price_path)
    output_path = Path(output_path)

    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    prices = pd.read_parquet(price_path)
    features = add_return_features(prices=prices, windows=windows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return features
