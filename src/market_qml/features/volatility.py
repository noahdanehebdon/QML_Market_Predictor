"""Rolling realized volatility feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

VOLATILITY_WINDOWS = [5, 20, 60]
TRADING_DAYS_PER_YEAR = 252
REQUIRED_RETURN_COLUMNS = {"symbol", "date", "return_1d"}


def add_volatility_features(
    features: pd.DataFrame,
    windows: list[int] | None = None,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Add annualized rolling realized volatility features by symbol.

    Volatility is the rolling standard deviation of daily returns through the
    current row, multiplied by sqrt(annualization_factor). No future returns are
    used.
    """
    windows = windows or VOLATILITY_WINDOWS
    missing_columns = REQUIRED_RETURN_COLUMNS - set(features.columns)
    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not windows:
        raise ValueError("At least one volatility window is required.")

    invalid_windows = [window for window in windows if window <= 1]
    if invalid_windows:
        raise ValueError(
            "Volatility windows must be integers greater than 1: "
            + ", ".join(str(window) for window in invalid_windows)
        )

    result = features.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["return_1d"] = pd.to_numeric(result["return_1d"], errors="coerce")

    if result["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")

    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    returns_by_symbol = result.groupby("symbol", sort=False)["return_1d"]

    for window in windows:
        rolling_std = returns_by_symbol.transform(
            lambda returns: returns.rolling(window=window, min_periods=window).std(
                ddof=0
            )
        )
        result[f"realized_vol_{window}d"] = rolling_std * (annualization_factor**0.5)

    return result


def build_price_volatility_features(
    feature_path: str | Path,
    output_path: str | Path,
    windows: list[int] | None = None,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Load return features, add volatility features, save, and return them."""
    feature_path = Path(feature_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Price return feature file not found: {feature_path}. "
            "Run python -m scripts.build_price_return_features first."
        )

    features = pd.read_parquet(feature_path)
    result = add_volatility_features(
        features=features,
        windows=windows,
        annualization_factor=annualization_factor,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result
