"""
Build a daily, market-aligned macro feature table.

Inputs:
- data/processed/macro.parquet
- processed equity price/features parquet file

Output:
- data/processed/macro_daily.parquet

Example usage:

    python -m scripts.build_macro_daily --prices data/processed/price_features.parquet

Optional conservative mode for daily rates:

    python -m scripts.build_macro_daily --prices data/processed/price_features.parquet --lag-daily-rates
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MACRO_PATH = Path("data/processed/macro.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/macro_daily.parquet")


DAILY_RATE_COLUMNS = [
    "treasury_10y",
    "treasury_2y",
    "fed_funds",
]

MONTHLY_MACRO_COLUMNS = [
    "cpi_all_items_sa",
    "unemployment_rate",
    "industrial_production",
]

EXPECTED_COLUMNS = DAILY_RATE_COLUMNS + MONTHLY_MACRO_COLUMNS


def load_trading_dates(price_path: Path) -> pd.DatetimeIndex:
    """
    Load unique trading dates from a processed price/features table.

    Handles common layouts:
    - date column
    - DatetimeIndex
    - MultiIndex with a date level
    """
    if not price_path.exists():
        raise FileNotFoundError(f"Price file not found: {price_path}")

    prices = pd.read_parquet(price_path)

    # Case 1: explicit date column
    if "date" in prices.columns:
        dates = pd.to_datetime(prices["date"], errors="coerce")

    # Case 2: MultiIndex with a date-like level
    elif isinstance(prices.index, pd.MultiIndex):
        index_names = list(prices.index.names)

        if "date" in index_names:
            dates = pd.to_datetime(
                prices.index.get_level_values("date"),
                errors="coerce",
            )
        else:
            # Try each index level and use the first one that parses well as dates.
            dates = None

            for level_number in range(prices.index.nlevels):
                candidate = pd.to_datetime(
                    prices.index.get_level_values(level_number),
                    errors="coerce",
                )

                if candidate.notna().mean() > 0.95:
                    dates = candidate
                    break

            if dates is None:
                raise ValueError(
                    "Could not identify a date level in the price file MultiIndex. "
                    "Expected a 'date' index level or a date column."
                )

    # Case 3: simple DatetimeIndex
    else:
        dates = pd.to_datetime(prices.index, errors="coerce")

    dates = pd.DatetimeIndex(dates).dropna()
    dates = dates.normalize().unique().sort_values()

    if len(dates) == 0:
        raise ValueError(f"No valid trading dates found in {price_path}")

    return dates


def load_macro(macro_path: Path) -> pd.DataFrame:
    """Load source-aligned macro data."""
    if not macro_path.exists():
        raise FileNotFoundError(f"Macro file not found: {macro_path}")

    macro = pd.read_parquet(macro_path)
    macro.index = pd.to_datetime(macro.index, errors="coerce")
    macro = macro[macro.index.notna()]
    macro.index = macro.index.normalize()
    macro = macro.sort_index()

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in macro.columns]

    if missing_columns:
        raise ValueError(
            "Macro file is missing expected columns: "
            + ", ".join(missing_columns)
        )

    return macro[EXPECTED_COLUMNS]


def merge_asof_to_trading_dates(
    trading_dates: pd.DatetimeIndex,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backward as-of merge observations onto trading dates.

    Each trading date receives the most recent macro observation whose available
    date is less than or equal to that trading date.
    """
    left = pd.DataFrame({"date": trading_dates}).sort_values("date")

    right = observations.copy()
    right = right.sort_index()
    right = right.reset_index(names="date")
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    right = right.sort_values("date")

    merged = pd.merge_asof(
        left,
        right,
        on="date",
        direction="backward",
    )

    merged = merged.set_index("date")
    merged.index.name = "date"

    return merged


def build_daily_rate_features(
    macro: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    lag_daily_rates: bool,
) -> pd.DataFrame:
    """
    Align daily Fed rate series to market trading dates.

    By default, same-date rate observations are allowed. This is reasonable if
    the feature table is used after market close or for next-period prediction.

    If --lag-daily-rates is passed, rate observations are shifted forward by
    one calendar day before alignment, making them available only after the
    observation date.
    """
    daily_rates = macro[DAILY_RATE_COLUMNS].dropna(how="all")

    if lag_daily_rates:
        daily_rates = daily_rates.copy()
        daily_rates.index = daily_rates.index + pd.Timedelta(days=1)

    daily_aligned = merge_asof_to_trading_dates(
        trading_dates=trading_dates,
        observations=daily_rates,
    )

    return daily_aligned[DAILY_RATE_COLUMNS]


def build_monthly_macro_features(
    macro: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align monthly macro series to market trading dates without look-ahead bias.

    First implementation:
    - Treat monthly macro observations as available one month after the
      observation month.
    - Example: January CPI dated 2026-01-01 becomes available on 2026-02-01.
    - Then forward-fill the latest available value to trading dates.

    This is intentionally conservative and avoids using same-month macro values
    before they would have been known.
    """
    monthly = macro[MONTHLY_MACRO_COLUMNS].dropna(how="all").copy()

    # Move observation dates forward by one month to represent safe availability.
    monthly.index = monthly.index + pd.DateOffset(months=1)
    monthly.index = pd.to_datetime(monthly.index).normalize()

    monthly_aligned = merge_asof_to_trading_dates(
        trading_dates=trading_dates,
        observations=monthly,
    )

    return monthly_aligned[MONTHLY_MACRO_COLUMNS]


def build_macro_daily(
    price_path: Path,
    macro_path: Path,
    output_path: Path,
    lag_daily_rates: bool,
) -> pd.DataFrame:
    """Build and save the market-aligned daily macro table."""
    trading_dates = load_trading_dates(price_path)
    macro = load_macro(macro_path)

    daily_rates = build_daily_rate_features(
        macro=macro,
        trading_dates=trading_dates,
        lag_daily_rates=lag_daily_rates,
    )

    monthly_macro = build_monthly_macro_features(
        macro=macro,
        trading_dates=trading_dates,
    )

    macro_daily = pd.concat([daily_rates, monthly_macro], axis=1)
    macro_daily = macro_daily[EXPECTED_COLUMNS]
    macro_daily = macro_daily.sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro_daily.to_parquet(output_path)

    return macro_daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build daily market-aligned macro features."
    )

    parser.add_argument(
        "--prices",
        type=Path,
        required=True,
        help="Path to processed price/features parquet file containing trading dates.",
    )

    parser.add_argument(
        "--macro",
        type=Path,
        default=DEFAULT_MACRO_PATH,
        help="Path to processed source-aligned macro parquet file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save daily market-aligned macro parquet file.",
    )

    parser.add_argument(
        "--lag-daily-rates",
        action="store_true",
        help=(
            "Shift daily rate observations forward one day before alignment. "
            "Use this for a more conservative no-look-ahead setup."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    macro_daily = build_macro_daily(
        price_path=args.prices,
        macro_path=args.macro,
        output_path=args.output,
        lag_daily_rates=args.lag_daily_rates,
    )

    print(f"Saved daily macro table to {args.output}")
    print(f"Rows: {len(macro_daily)}")
    print("\nColumns:")
    print(list(macro_daily.columns))

    print("\nDate range:")
    print(macro_daily.index.min(), "to", macro_daily.index.max())

    print("\nTail:")
    print(macro_daily.tail())

    print("\nMissing values by column:")
    print(macro_daily.isna().sum())


if __name__ == "__main__":
    main()