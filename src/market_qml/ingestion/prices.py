"""Alpaca historical daily price ingestion.

This module fetches historical OHLCV bars from Alpaca, handles pagination,
normalizes the returned JSON, and saves data for downstream feature engineering.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ALPACA_STOCK_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


@dataclass(frozen=True)
class PriceRequest:
    symbols: list[str]
    start: str
    end: str | None = None
    timeframe: str = "1Day"
    adjustment: str = "all"
    feed: str | None = None
    limit: int = 10_000


def _credentials() -> tuple[str, str]:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY in your .env file or environment."
        )

    return api_key, secret_key


def _headers() -> dict[str, str]:
    api_key, secret_key = _credentials()
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def _normalize_datetime(value: str | None) -> str | None:
    """Convert YYYY-MM-DD to a simple RFC3339-style timestamp.

    Alpaca accepts timestamp parameters. For daily bars, this keeps config
    readable while still sending explicit timestamps.
    """
    if value is None:
        return None

    if "T" in value:
        return value

    return f"{value}T00:00:00Z"


def _request_page(
    session: requests.Session,
    params: dict[str, Any],
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    """Request one page of Alpaca bars with basic retry handling."""
    for attempt in range(1, max_retries + 1):
        response = session.get(
            ALPACA_STOCK_BARS_URL,
            headers=_headers(),
            params=params,
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()

        # Retry common temporary failures.
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            time.sleep(sleep_seconds * attempt)
            continue

        raise RuntimeError(
            "Alpaca request failed. "
            f"Status={response.status_code}. "
            f"Response={response.text[:500]}"
        )

    raise RuntimeError("Alpaca request failed after retries.")


def _normalize_bar_pages(pages: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize Alpaca bar pages into a tidy DataFrame.

    Expected Alpaca shape:
    {
        "bars": {
            "AAPL": [
                {"t": "...", "o": ..., "h": ..., "l": ..., "c": ..., "v": ..., "n": ..., "vw": ...}
            ]
        },
        "next_page_token": "..."
    }
    """
    rows: list[dict[str, Any]] = []

    for page in pages:
        bars_by_symbol = page.get("bars", {})

        for symbol, bars in bars_by_symbol.items():
            for bar in bars:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar.get("t"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "trade_count": bar.get("n"),
                        "vwap": bar.get("vw"),
                    }
                )

    columns = [
        "symbol",
        "timestamp",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    df = df[
        [
            "symbol",
            "timestamp",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
    ]

    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def fetch_alpaca_bars(request: PriceRequest) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fetch all paginated historical bars for the requested symbols."""
    if not request.symbols:
        raise ValueError("PriceRequest.symbols cannot be empty.")

    params: dict[str, Any] = {
        "symbols": ",".join(request.symbols),
        "timeframe": request.timeframe,
        "start": _normalize_datetime(request.start),
        "adjustment": request.adjustment,
        "limit": request.limit,
    }

    end = _normalize_datetime(request.end)
    if end is not None:
        params["end"] = end

    if request.feed:
        params["feed"] = request.feed

    pages: list[dict[str, Any]] = []
    session = requests.Session()
    page_token: str | None = None

    while True:
        if page_token:
            params["page_token"] = page_token
        elif "page_token" in params:
            del params["page_token"]

        page = _request_page(session=session, params=params)
        pages.append(page)

        page_token = page.get("next_page_token")
        if not page_token:
            break

    df = _normalize_bar_pages(pages)
    return df, pages


def save_raw_pages(pages: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)


def save_raw_bars(pages: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_bar_pages(pages).to_parquet(output_path, index=False)


def save_prices(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
