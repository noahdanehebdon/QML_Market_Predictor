"""Command-line script for Issue 01: Alpaca daily OHLCV ingestion."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from market_qml.ingestion.prices import (
    PriceRequest,
    fetch_alpaca_bars,
    save_prices,
    save_raw_bars,
)

LOGGER = logging.getLogger(__name__)


def load_candidate_symbols(
    asset_history_path: Path,
    *,
    exchanges: list[str],
    limit: int,
    benchmark: str,
) -> list[str]:
    """Select a stable broad candidate pool from the latest archived snapshot."""
    if limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    assets = pd.read_parquet(asset_history_path)
    required = {
        "symbol",
        "effective_date",
        "asset_class",
        "exchange",
        "status",
        "tradable",
        "security_type",
    }
    missing = required - set(assets)
    if missing:
        raise ValueError("Asset history is missing: " + ", ".join(sorted(missing)))
    assets["effective_date"] = pd.to_datetime(assets["effective_date"]).dt.normalize()
    latest_date = assets["effective_date"].max()
    latest = assets.loc[assets["effective_date"].eq(latest_date)].copy()
    latest = latest.loc[
        latest["asset_class"].eq("us_equity")
        & latest["exchange"].isin(exchanges)
        & latest["status"].eq("active")
        & latest["tradable"].fillna(False)
        & latest["security_type"].eq("common_stock")
    ]
    latest["selection_key"] = (
        latest["symbol"]
        .astype(str)
        .map(lambda symbol: hashlib.sha256(symbol.encode("utf-8")).hexdigest())
    )
    symbols = (
        latest.sort_values("selection_key")["symbol"].astype(str).head(limit).tolist()
    )
    benchmark = benchmark.upper()
    if benchmark not in symbols:
        symbols.append(benchmark)
    return symbols


def merge_price_history(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Preserve removed symbols while replacing overlapping bars with fresh data."""
    required = {"symbol", "timestamp"}
    for name, frame in [("Existing prices", existing), ("Fresh prices", fresh)]:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} are missing: " + ", ".join(sorted(missing)))
    combined = pd.concat([existing, fresh], ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return (
        combined.drop_duplicates(["symbol", "timestamp"], keep="last")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


def _clean_feed(value: object) -> str | None:
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str.lower() in {"", "none", "null"}:
        return None

    return value_str


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    config_path = Path("configs/universe.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    universe = config["universe"]
    dates = config["dates"]
    alpaca = config.get("alpaca", {})

    symbols = list(universe["symbols"])
    benchmark = universe["benchmark"]

    point_in_time = config.get("point_in_time", {})
    asset_history_path = Path(
        point_in_time.get("asset_history_path", "data/processed/asset_history.parquet")
    )
    if point_in_time.get("enabled") and asset_history_path.exists():
        symbols = load_candidate_symbols(
            asset_history_path,
            exchanges=point_in_time["candidate_exchanges"],
            limit=int(point_in_time["candidate_limit"]),
            benchmark=benchmark,
        )
        LOGGER.info(
            "Using prospective candidate pool from latest private asset snapshot."
        )
        symbols = sorted(
            set(symbols) | {str(symbol).upper() for symbol in universe["symbols"]}
        )
    elif point_in_time.get("enabled"):
        LOGGER.warning(
            "No point-in-time asset history exists yet; using the legacy seed universe."
        )

    if benchmark not in symbols:
        symbols.append(benchmark)

    request = PriceRequest(
        symbols=symbols,
        start=dates["start"],
        end=dates.get("end"),
        timeframe=alpaca.get("timeframe", "1Day"),
        adjustment=alpaca.get("adjustment", "all"),
        feed=_clean_feed(alpaca.get("feed")),
    )

    LOGGER.info("Fetching Alpaca bars for %s symbols.", len(request.symbols))
    LOGGER.info("Symbols: %s", ", ".join(request.symbols))
    LOGGER.info("Start: %s", request.start)
    LOGGER.info("End: %s", request.end)
    LOGGER.info("Feed: %s", request.feed or "account default")

    prices, raw_pages = fetch_alpaca_bars(request)

    if prices.empty:
        LOGGER.warning(
            "No bars returned. Check symbols, date range, feed, and account permissions."
        )
        return

    raw_path = Path("data/raw/alpaca_bars.parquet")
    processed_path = Path("data/processed/prices.parquet")

    save_raw_bars(raw_pages, raw_path)
    if point_in_time.get("enabled") and processed_path.exists():
        prices = merge_price_history(pd.read_parquet(processed_path), prices)
    save_prices(prices, processed_path)

    LOGGER.info("Saved raw bars to: %s", raw_path)
    LOGGER.info("Saved normalized prices to: %s", processed_path)
    LOGGER.info("Rows: %s", len(prices))
    LOGGER.info("Symbols: %s", sorted(prices["symbol"].unique()))


if __name__ == "__main__":
    main()
