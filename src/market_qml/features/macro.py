"""Macroeconomic feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RATE_CHANGE_WINDOWS = [5, 20, 60]
MACRO_CHANGE_WINDOWS = [21, 63, 252]
REQUIRED_MARKET_COLUMNS = {"symbol", "date"}
REQUIRED_MACRO_COLUMNS = {
    "treasury_10y",
    "treasury_2y",
    "fed_funds",
    "cpi_all_items_sa",
    "unemployment_rate",
    "industrial_production",
}


def add_macro_features(
    macro_daily: pd.DataFrame,
    rate_change_windows: list[int] | None = None,
    macro_change_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create leakage-safe macro features from a daily-aligned macro table."""
    rate_change_windows = rate_change_windows or RATE_CHANGE_WINDOWS
    macro_change_windows = macro_change_windows or MACRO_CHANGE_WINDOWS

    macro = _normalize_macro_daily(macro_daily)
    missing_columns = REQUIRED_MACRO_COLUMNS - set(macro.columns)
    if missing_columns:
        raise ValueError(
            "Macro table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    _validate_windows(rate_change_windows, "rate_change_windows")
    _validate_windows(macro_change_windows, "macro_change_windows")

    result = macro.copy()
    for column in REQUIRED_MACRO_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["yield_spread_10y_2y"] = result["treasury_10y"] - result["treasury_2y"]

    rate_columns = ["treasury_10y", "treasury_2y", "fed_funds", "yield_spread_10y_2y"]
    for window in rate_change_windows:
        for column in rate_columns:
            result[f"{column}_change_{window}d"] = result[column].diff(periods=window)

    for window in macro_change_windows:
        result[f"cpi_inflation_{window}d"] = result["cpi_all_items_sa"].pct_change(
            periods=window,
            fill_method=None,
        )
        result[f"unemployment_rate_change_{window}d"] = result[
            "unemployment_rate"
        ].diff(periods=window)
        result[f"industrial_production_growth_{window}d"] = result[
            "industrial_production"
        ].pct_change(periods=window, fill_method=None)

    return result.reset_index()


def merge_macro_features(
    market_features: pd.DataFrame,
    macro_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge daily macro features into a symbol/date market feature table."""
    missing_columns = REQUIRED_MARKET_COLUMNS - set(market_features.columns)
    if missing_columns:
        raise ValueError(
            "Market feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if "date" not in macro_features.columns:
        raise ValueError("Macro feature table is missing required column: date")

    market = market_features.copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()

    macro = macro_features.copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()

    if market["date"].isna().any():
        raise ValueError("Market feature table contains invalid dates.")

    if macro["date"].isna().any():
        raise ValueError("Macro feature table contains invalid dates.")

    if macro["date"].duplicated().any():
        raise ValueError("Macro feature table contains duplicate dates.")

    result = market.merge(macro, on="date", how="left", validate="many_to_one")
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def build_macro_feature_table(
    feature_path: str | Path,
    macro_daily_path: str | Path,
    output_path: str | Path,
    rate_change_windows: list[int] | None = None,
    macro_change_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Load market and macro tables, merge macro features, save, and return."""
    feature_path = Path(feature_path)
    macro_daily_path = Path(macro_daily_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Market feature file not found: {feature_path}. "
            "Run python -m scripts.build_benchmark_relative_features first."
        )

    if not macro_daily_path.exists():
        raise FileNotFoundError(
            f"Daily macro file not found: {macro_daily_path}. "
            "Run python -m scripts.build_macro_daily first."
        )

    market_features = pd.read_parquet(feature_path)
    macro_daily = pd.read_parquet(macro_daily_path)
    macro_features = add_macro_features(
        macro_daily=macro_daily,
        rate_change_windows=rate_change_windows,
        macro_change_windows=macro_change_windows,
    )
    result = merge_macro_features(
        market_features=market_features,
        macro_features=macro_features,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result


def _normalize_macro_daily(macro_daily: pd.DataFrame) -> pd.DataFrame:
    macro = macro_daily.copy()

    if "date" in macro.columns:
        macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
        macro = macro.set_index("date")
    else:
        macro.index = pd.to_datetime(macro.index, errors="coerce")
        macro.index = macro.index.normalize()
        macro.index.name = "date"

    macro = macro[macro.index.notna()]
    macro = macro.sort_index()

    if macro.index.duplicated().any():
        raise ValueError("Macro table contains duplicate dates.")

    return macro


def _validate_windows(windows: list[int], name: str) -> None:
    if not windows:
        raise ValueError(f"{name} must contain at least one window.")

    invalid_windows = [window for window in windows if window <= 0]
    if invalid_windows:
        raise ValueError(
            f"{name} must contain positive integers: "
            + ", ".join(str(window) for window in invalid_windows)
        )
