"""Benchmark-relative feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BENCHMARK_WINDOWS = [20, 60]
EXCESS_RETURN_WINDOWS = [1, 5, 20, 60]
REQUIRED_BENCHMARK_COLUMNS = {"symbol", "date", "return_1d"}


def add_benchmark_relative_features(
    features: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    windows: list[int] | None = None,
    excess_return_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add benchmark-relative features to a cumulative price feature table."""
    windows = windows or BENCHMARK_WINDOWS
    excess_return_windows = excess_return_windows or EXCESS_RETURN_WINDOWS

    missing_columns = REQUIRED_BENCHMARK_COLUMNS - set(features.columns)
    if missing_columns:
        raise ValueError(
            "Feature table is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if not windows:
        raise ValueError("At least one benchmark window is required.")

    invalid_windows = [window for window in windows if window <= 1]
    if invalid_windows:
        raise ValueError(
            "Benchmark windows must be integers greater than 1: "
            + ", ".join(str(window) for window in invalid_windows)
        )

    result = features.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["return_1d"] = pd.to_numeric(result["return_1d"], errors="coerce")

    if result["date"].isna().any():
        raise ValueError("Feature table contains invalid dates.")

    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    benchmark_symbol = benchmark_symbol.upper()
    benchmark = _benchmark_frame(result, benchmark_symbol)

    result = result.merge(benchmark, on="date", how="left", validate="many_to_one")
    if result["benchmark_return_1d"].isna().all():
        raise ValueError(f"No benchmark rows found for symbol {benchmark_symbol}.")

    for window in excess_return_windows:
        stock_column = f"return_{window}d"
        benchmark_column = f"benchmark_return_{window}d"
        if stock_column in result.columns and benchmark_column in result.columns:
            result[f"excess_return_{window}d_vs_{benchmark_symbol.lower()}"] = (
                result[stock_column] - result[benchmark_column]
            )

    grouped = result.groupby("symbol", sort=False, group_keys=False)

    for window in windows:
        result[f"rolling_corr_{window}d_vs_{benchmark_symbol.lower()}"] = result.groupby(
            "symbol",
            sort=False,
            group_keys=False,
        )[["return_1d", "benchmark_return_1d"]].apply(
            lambda group: group["return_1d"].rolling(
                window=window,
                min_periods=window,
            ).corr(group["benchmark_return_1d"])
        )

        rolling_cov = result.groupby(
            "symbol",
            sort=False,
            group_keys=False,
        )[["return_1d", "benchmark_return_1d"]].apply(
            lambda group: group["return_1d"].rolling(
                window=window,
                min_periods=window,
            ).cov(group["benchmark_return_1d"], ddof=0)
        )
        benchmark_variance = grouped["benchmark_return_1d"].transform(
            lambda returns: returns.rolling(
                window=window,
                min_periods=window,
            ).var(ddof=0)
        )
        result[f"rolling_beta_{window}d_vs_{benchmark_symbol.lower()}"] = (
            rolling_cov / benchmark_variance
        )

        stock_vol_column = f"realized_vol_{window}d"
        benchmark_vol_column = f"benchmark_realized_vol_{window}d"
        if stock_vol_column in result.columns and benchmark_vol_column in result.columns:
            result[f"relative_vol_{window}d_vs_{benchmark_symbol.lower()}"] = (
                result[stock_vol_column] / result[benchmark_vol_column]
            )

        stock_return_column = f"return_{window}d"
        benchmark_return_column = f"benchmark_return_{window}d"
        if stock_return_column in result.columns and benchmark_return_column in result.columns:
            result[f"relative_momentum_{window}d_vs_{benchmark_symbol.lower()}"] = (
                result[stock_return_column] - result[benchmark_return_column]
            )

    result = result.drop(columns=[column for column in result.columns if column.startswith("benchmark_")])
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def _benchmark_frame(features: pd.DataFrame, benchmark_symbol: str) -> pd.DataFrame:
    benchmark = features[features["symbol"] == benchmark_symbol].copy()
    if benchmark.empty:
        raise ValueError(f"No benchmark rows found for symbol {benchmark_symbol}.")

    if benchmark["date"].duplicated().any():
        raise ValueError(f"Benchmark symbol {benchmark_symbol} has duplicate dates.")

    benchmark_columns = ["date", "return_1d"]
    benchmark_columns.extend(
        column
        for column in features.columns
        if column.startswith("return_") and column != "return_1d"
    )
    benchmark_columns.extend(
        column for column in features.columns if column.startswith("realized_vol_")
    )

    benchmark = benchmark[benchmark_columns]
    benchmark = benchmark.rename(
        columns={
            column: f"benchmark_{column}"
            for column in benchmark_columns
            if column != "date"
        }
    )
    return benchmark


def build_benchmark_relative_features(
    feature_path: str | Path,
    output_path: str | Path,
    benchmark_symbol: str = "SPY",
    windows: list[int] | None = None,
    excess_return_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Load price features, add benchmark-relative features, save, and return."""
    feature_path = Path(feature_path)
    output_path = Path(output_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Price feature file not found: {feature_path}. "
            "Run python -m scripts.build_price_volume_features first."
        )

    features = pd.read_parquet(feature_path)
    result = add_benchmark_relative_features(
        features=features,
        benchmark_symbol=benchmark_symbol,
        windows=windows,
        excess_return_windows=excess_return_windows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    return result
