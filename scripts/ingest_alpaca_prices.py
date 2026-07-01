"""Command-line script for Issue 01: Alpaca daily OHLCV ingestion."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv

from market_qml.ingestion.prices import (
    PriceRequest,
    fetch_alpaca_bars,
    save_prices,
    save_raw_pages,
)


def _clean_feed(value: object) -> str | None:
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str.lower() in {"", "none", "null"}:
        return None

    return value_str


def main() -> None:
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

    print(f"Fetching Alpaca bars for {len(request.symbols)} symbols...")
    print(f"Symbols: {', '.join(request.symbols)}")
    print(f"Start: {request.start}")
    print(f"End: {request.end}")
    print(f"Feed: {request.feed or 'account default'}")

    prices, raw_pages = fetch_alpaca_bars(request)

    if prices.empty:
        print("No bars returned. Check symbols, date range, feed, and account permissions.")
        return

    raw_path = Path("data/raw/alpaca_bars_raw.json")
    processed_path = Path("data/processed/prices.parquet")

    save_raw_pages(raw_pages, raw_path)
    save_prices(prices, processed_path)

    print(f"Saved raw response pages to: {raw_path}")
    print(f"Saved normalized prices to: {processed_path}")
    print()
    print("Preview:")
    print(prices.head())
    print()
    print("Rows:", len(prices))
    print("Symbols:", sorted(prices["symbol"].unique()))


if __name__ == "__main__":
    main()