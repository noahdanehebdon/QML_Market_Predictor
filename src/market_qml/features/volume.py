"""Volume and liquidity feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

VOLUME_WINDOWS = [5, 20, 60]
REQUIRED_VOLUME_COLUMNS = {"symbol", "date", "close", "volume"}


def add_volume_features(
    features: pd.DataFrame,
    windows: list[int] | None = None,
    liquidity_min_avg_dollar_volume: float | None = None,
    liquidity_window: int = 20,
) -> pd.DataFrame:
    """Add rolling volume, dollar-volume, and volume-shock features by symbol."""
    windows = windows or VOLUME_WINDOWS
    missing_columns = REQUIRED_VOLUME_COLUMNS - set(features.columns)
    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not windows:
        raise ValueError("At least one volume window is required.")

    invalid_windows = [window for window in windows if window <= 1]
    if invalid_windows:
        raise ValueError(
            "Volume windows must be integers greater than 1: "
            + ", ".join(str(window) for window in invalid_windows)
        )

    if liquidity_min_avg_dollar_volume is not None and liquidity_window not in windows:
        raise ValueError(
            "liquidity_window must be one of the configured volume windows."
        )

    result = features.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["volume"] = pd.to_numeric(result["volume"], errors="coerce")

    if result["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")

    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    result["dollar_volume"] = result["close"] * result["volume"]

    grouped = result.groupby("symbol", sort=False)

    for window in windows:
        result[f"avg_volume_{window}d"] = grouped["volume"].transform(
            lambda volume: volume.rolling(window=window, min_periods=window).mean()
        )
        prior_avg_volume = grouped["volume"].transform(
            lambda volume: (
                volume.shift(1).rolling(window=window, min_periods=window).mean()
            )
        )
        result[f"volume_shock_{window}d"] = (result["volume"] / prior_avg_volume) - 1
        result[f"avg_dollar_volume_{window}d"] = grouped["dollar_volume"].transform(
            lambda dollar_volume: dollar_volume.rolling(
                window=window, min_periods=window
            ).mean()
        )

    if liquidity_min_avg_dollar_volume is not None:
        column = f"avg_dollar_volume_{liquidity_window}d"
        result[f"is_liquid_{liquidity_window}d"] = (
            result[column] >= liquidity_min_avg_dollar_volume
        )

    return result


def build_price_volume_features(
    feature_path: str | Path,
    output_path: str | Path,
    windows: list[int] | None = None,
    liquidity_min_avg_dollar_volume: float | None = None,
    liquidity_window: int = 20,
) -> pd.DataFrame:
    """Load price features, add volume features, save, and return them."""
    feature_path = Path(feature_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Price feature file not found: {feature_path}. "
            "Run python -m scripts.build_price_volatility_features first."
        )

    features = pd.read_parquet(feature_path)
    result = add_volume_features(
        features=features,
        windows=windows,
        liquidity_min_avg_dollar_volume=liquidity_min_avg_dollar_volume,
        liquidity_window=liquidity_window,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result
