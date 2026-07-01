"""Command-line script for Issue 01: Alpaca daily OHLCV ingestion."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from market_qml.ingestion.prices import (
    PriceRequest,
    fetch_alpaca_bars,
    save_prices,
    save_raw_bars,
)

LOGGER = logging.getLogger(__name__)


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

    symbols = universe["symbols"]
    benchmark = universe["benchmark"]

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
        LOGGER.warning("No bars returned. Check symbols, date range, feed, and account permissions.")
        return

    raw_path = Path("data/raw/alpaca_bars.parquet")
    processed_path = Path("data/processed/prices.parquet")

    save_raw_bars(raw_pages, raw_path)
    save_prices(prices, processed_path)

    LOGGER.info("Saved raw bars to: %s", raw_path)
    LOGGER.info("Saved normalized prices to: %s", processed_path)
    LOGGER.info("Rows: %s", len(prices))
    LOGGER.info("Symbols: %s", sorted(prices["symbol"].unique()))


if __name__ == "__main__":
    main()
