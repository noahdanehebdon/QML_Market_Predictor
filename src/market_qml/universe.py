"""Point-in-time liquid-equity universe construction and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = {"symbol", "date", "close", "volume"}
REQUIRED_ASSET_COLUMNS = {
    "symbol",
    "effective_date",
    "asset_class",
    "status",
    "tradable",
}


@dataclass(frozen=True)
class UniverseRules:
    min_price: float = 5.0
    min_median_dollar_volume: float = 5_000_000.0
    liquidity_window: int = 20
    min_history_days: int = 252
    min_names: int = 100
    min_sectors: int = 8
    min_sector_names: int = 3
    allowed_security_types: tuple[str, ...] = ("common_stock",)


def build_point_in_time_universe(
    prices: pd.DataFrame,
    asset_history: pd.DataFrame,
    *,
    metadata_history: pd.DataFrame | None = None,
    rules: UniverseRules = UniverseRules(),
    benchmark_symbol: str = "SPY",
    legacy_seed_symbols: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Return a complete date/symbol panel with trailing-only membership flags."""
    _require_columns(prices, REQUIRED_PRICE_COLUMNS, "Prices")
    _require_columns(asset_history, REQUIRED_ASSET_COLUMNS, "Asset history")
    _validate_rules(rules)
    benchmark_symbol = benchmark_symbol.strip().upper()

    normalized_prices = prices.copy()
    normalized_prices["symbol"] = normalized_prices["symbol"].astype(str).str.upper()
    normalized_prices["date"] = (
        pd.to_datetime(normalized_prices["date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    normalized_prices["close"] = pd.to_numeric(
        normalized_prices["close"], errors="coerce"
    )
    normalized_prices["volume"] = pd.to_numeric(
        normalized_prices["volume"], errors="coerce"
    )
    if normalized_prices["date"].isna().any():
        raise ValueError("Prices contain invalid dates.")
    if normalized_prices.duplicated(["symbol", "date"]).any():
        raise ValueError("Prices contain duplicate symbol/date rows.")

    # Asset snapshots cover the full Alpaca security master. Membership can only
    # be evaluated for the bounded candidate pool that price ingestion selected;
    # crossing every asset with every date can create tens of millions of rows.
    symbols = sorted(normalized_prices["symbol"].unique())
    assets = _normalize_effective_history(asset_history, "Asset history")
    assets = assets.loc[assets["symbol"].isin(symbols)].reset_index(drop=True)
    dates = pd.DatetimeIndex(normalized_prices["date"].unique()).sort_values()
    panel = pd.MultiIndex.from_product(
        [symbols, dates], names=["symbol", "date"]
    ).to_frame(index=False)
    panel = panel.merge(normalized_prices, on=["symbol", "date"], how="left")
    panel["has_price"] = panel["close"].notna() & panel["volume"].notna()
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["history_days"] = panel.groupby("symbol", sort=False)["has_price"].cumsum()
    panel["first_observed_date"] = panel["date"].where(panel["has_price"])
    panel["first_observed_date"] = panel.groupby("symbol", sort=False)[
        "first_observed_date"
    ].transform("min")
    panel["last_observed_through_date"] = panel["date"].where(panel["has_price"])
    panel["last_observed_through_date"] = panel.groupby("symbol", sort=False)[
        "last_observed_through_date"
    ].ffill()
    panel["trailing_median_dollar_volume"] = panel.groupby("symbol", sort=False)[
        "dollar_volume"
    ].transform(
        lambda values: values.rolling(
            rules.liquidity_window, min_periods=rules.liquidity_window
        ).median()
    )
    panel = _merge_effective_history(panel, assets)
    panel = panel.rename(columns={"effective_date": "asset_effective_date"})

    if metadata_history is not None:
        metadata = _normalize_effective_history(metadata_history, "Metadata history")
        metadata = metadata.loc[metadata["symbol"].isin(symbols)].reset_index(drop=True)
        panel = _merge_effective_history(panel, metadata)
        panel = panel.rename(columns={"effective_date": "metadata_effective_date"})
    # Retain the original public column while preventing metadata merges from
    # obscuring which effective date controls tradability.
    panel["effective_date"] = panel["asset_effective_date"]
    for column in ["sector", "industry", "market_cap"]:
        if column not in panel:
            panel[column] = pd.NA
    panel["market_cap"] = pd.to_numeric(panel["market_cap"], errors="coerce")

    panel["eligible_price"] = panel["has_price"] & (panel["close"] >= rules.min_price)
    panel["eligible_liquidity"] = panel["trailing_median_dollar_volume"] >= (
        rules.min_median_dollar_volume
    )
    panel["eligible_history"] = panel["history_days"] >= rules.min_history_days
    panel["eligible_tradability"] = (
        panel["asset_class"].eq("us_equity")
        & panel["status"].eq("active")
        & panel["tradable"].eq(True)
    )
    panel["eligible_security_type"] = (
        panel["security_type"].isin(rules.allowed_security_types)
        if "security_type" in panel
        else False
    )
    first_snapshot = assets["effective_date"].min()
    legacy_symbols = {str(symbol).upper() for symbol in legacy_seed_symbols}
    panel["legacy_seed_period"] = panel["date"].lt(first_snapshot) & panel[
        "symbol"
    ].isin(legacy_symbols)
    panel.loc[panel["legacy_seed_period"], "eligible_tradability"] = True
    panel.loc[panel["legacy_seed_period"], "eligible_security_type"] = True
    panel["asset_state_known"] = panel["asset_effective_date"].notna()
    panel["membership_basis"] = "effective_dated_asset_state"
    panel.loc[panel["legacy_seed_period"], "membership_basis"] = (
        "legacy_static_seed_survivorship_limited"
    )
    panel["is_benchmark"] = panel["symbol"].eq(benchmark_symbol)
    panel["is_member"] = (
        panel["eligible_price"]
        & panel["eligible_liquidity"]
        & panel["eligible_history"]
        & panel["eligible_tradability"]
        & panel["eligible_security_type"]
        & ~panel["is_benchmark"]
    )
    panel["exclusion_reason"] = _exclusion_reason(panel)
    panel["size_bucket"] = pd.NA
    member_market_caps = panel["is_member"] & panel["market_cap"].notna()
    panel.loc[member_market_caps, "size_bucket"] = (
        panel.loc[member_market_caps]
        .groupby("date", sort=False)["market_cap"]
        .transform(_size_bucket)
    )
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def universe_diagnostics(
    membership: pd.DataFrame, *, rules: UniverseRules = UniverseRules()
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Report daily coverage, membership transitions, and aggregate limitations."""
    _require_columns(
        membership,
        {"symbol", "date", "is_member", "sector", "size_bucket", "has_price"},
        "Membership",
    )
    ordered = membership.sort_values(["symbol", "date"]).copy()
    previous = ordered.groupby("symbol", sort=False)["is_member"].shift(
        fill_value=False
    )
    ordered["entered"] = ordered["is_member"] & ~previous
    ordered["exited"] = ~ordered["is_member"] & previous
    transitions = ordered.loc[
        ordered["entered"] | ordered["exited"],
        [
            "date",
            "symbol",
            "entered",
            "exited",
            "sector",
            "size_bucket",
            "exclusion_reason",
            "membership_basis",
            "status",
            "tradable",
        ],
    ].reset_index(drop=True)

    daily = (
        ordered.groupby("date", sort=True)
        .agg(
            observed_names=("has_price", "sum"),
            member_count=("is_member", "sum"),
            entries=("entered", "sum"),
            exits=("exited", "sum"),
            known_asset_states=("asset_state_known", "sum"),
            legacy_rows=("legacy_seed_period", "sum"),
        )
        .reset_index()
    )
    member_rows = ordered.loc[ordered["is_member"]]
    sector_counts = member_rows.groupby("date")["sector"].nunique(dropna=True)
    size_counts = member_rows.groupby("date")["size_bucket"].nunique(dropna=True)
    smallest_sector = (
        member_rows.groupby(["date", "sector"], dropna=True)
        .size()
        .groupby("date")
        .min()
    )
    daily["sector_count"] = daily["date"].map(sector_counts).fillna(0).astype(int)
    daily["size_bucket_count"] = daily["date"].map(size_counts).fillna(0).astype(int)
    daily["smallest_sector_names"] = (
        daily["date"].map(smallest_sector).fillna(0).astype(int)
    )
    previous_count = daily["member_count"].shift()
    daily["membership_turnover"] = (
        daily["entries"] + daily["exits"]
    ) / previous_count.replace(0, np.nan)
    daily["stable_deciles"] = daily["member_count"] >= max(10, rules.min_names)
    daily["stable_sector_controls"] = (daily["sector_count"] >= rules.min_sectors) & (
        daily["smallest_sector_names"] >= rules.min_sector_names
    )
    rows_per_date = ordered.groupby("date").size()
    daily["asset_state_coverage"] = daily["known_asset_states"] / daily["date"].map(
        rows_per_date
    )
    daily["legacy_row_share"] = daily["legacy_rows"] / daily["date"].map(rows_per_date)
    metadata_known = ordered["sector"].notna() | ordered["market_cap"].notna()
    metadata_counts = metadata_known.groupby(ordered["date"]).sum()
    daily["metadata_coverage"] = daily["date"].map(metadata_counts) / daily["date"].map(
        rows_per_date
    )
    inactive_exits = transitions["exited"] & (
        transitions["status"].ne("active") | transitions["tradable"].ne(True)
    )
    summary = {
        "start_date": daily["date"].min(),
        "end_date": daily["date"].max(),
        "median_member_count": float(daily["member_count"].median()),
        "minimum_member_count": int(daily["member_count"].min()),
        "mean_membership_turnover": float(daily["membership_turnover"].mean()),
        "entries": int(transitions["entered"].sum()),
        "exits": int(transitions["exited"].sum()),
        "stable_decile_date_share": float(daily["stable_deciles"].mean()),
        "stable_sector_date_share": float(daily["stable_sector_controls"].mean()),
        "asset_state_coverage": float(daily["asset_state_coverage"].mean()),
        "metadata_coverage": float(daily["metadata_coverage"].mean()),
        "legacy_row_share": float(daily["legacy_row_share"].mean()),
        "inactive_or_untradable_exits": int(inactive_exits.sum()),
        "prospective_start_date": ordered.loc[
            ordered["asset_state_known"], "date"
        ].min(),
    }
    return daily, transitions, summary


def _normalize_effective_history(data: pd.DataFrame, name: str) -> pd.DataFrame:
    result = data.copy()
    _require_columns(result, {"symbol", "effective_date"}, name)
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["effective_date"] = (
        pd.to_datetime(result["effective_date"], errors="coerce")
        .dt.normalize()
        .astype("datetime64[ns]")
    )
    if result["effective_date"].isna().any():
        raise ValueError(f"{name} contains invalid effective dates.")
    if result.duplicated(["symbol", "effective_date"]).any():
        raise ValueError(f"{name} contains duplicate symbol/effective_date rows.")
    return result.sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def _merge_effective_history(
    panel: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    pieces = []
    for symbol, symbol_panel in panel.groupby("symbol", sort=False):
        symbol_history = history.loc[history["symbol"].eq(symbol)].drop(
            columns="symbol"
        )
        merged = pd.merge_asof(
            symbol_panel.sort_values("date"),
            symbol_history.sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            direction="backward",
            allow_exact_matches=True,
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _size_bucket(values: pd.Series) -> pd.Series:
    percentile = values.rank(method="average", pct=True)
    return pd.cut(
        percentile,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["small", "mid", "large"],
        include_lowest=True,
    ).astype("string")


def _exclusion_reason(panel: pd.DataFrame) -> pd.Series:
    reason = pd.Series("eligible", index=panel.index, dtype="string")
    ordered_rules = [
        (panel["is_benchmark"], "benchmark"),
        (~panel["has_price"], "missing_price"),
        (~panel["eligible_tradability"], "inactive_or_untradable"),
        (~panel["eligible_security_type"], "unsupported_security_type"),
        (~panel["eligible_history"], "insufficient_history"),
        (~panel["eligible_liquidity"], "insufficient_liquidity"),
        (~panel["eligible_price"], "price_below_minimum"),
    ]
    for excluded, label in reversed(ordered_rules):
        reason.loc[excluded] = label
    return reason


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(data)
    if missing:
        raise ValueError(f"{name} are missing: " + ", ".join(sorted(missing)))


def _validate_rules(rules: UniverseRules) -> None:
    if rules.min_price <= 0 or rules.min_median_dollar_volume <= 0:
        raise ValueError("Price and dollar-volume thresholds must be positive.")
    for value in [
        rules.liquidity_window,
        rules.min_history_days,
        rules.min_names,
        rules.min_sectors,
        rules.min_sector_names,
    ]:
        if value <= 0:
            raise ValueError("Universe count and window rules must be positive.")
    if not rules.allowed_security_types:
        raise ValueError("At least one security type must be allowed.")
