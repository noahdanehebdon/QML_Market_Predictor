"""SEC ticker-to-CIK lookup helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _sec_user_agent() -> str:
    user_agent = os.getenv("SEC_USER_AGENT") or os.getenv("USER_AGENT")

    if not user_agent:
        raise RuntimeError(
            "Missing SEC user agent. Set SEC_USER_AGENT in your .env file or "
            "environment to a descriptive value with contact information."
        )

    return user_agent


def _headers(user_agent: str | None = None) -> dict[str, str]:
    return {
        "User-Agent": user_agent or _sec_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def _normalize_ticker(ticker: object) -> str:
    ticker_str = str(ticker).strip().upper()

    if not ticker_str:
        raise ValueError("Ticker cannot be empty.")

    return ticker_str


def format_cik(cik: object) -> str:
    """Return the SEC's standard 10-digit zero-padded CIK string."""
    try:
        cik_int = int(cik)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CIK value: {cik!r}") from exc

    if cik_int < 0:
        raise ValueError(f"Invalid CIK value: {cik!r}")

    return f"{cik_int:010d}"


def normalize_company_tickers(payload: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize the SEC company tickers payload into a tidy DataFrame."""
    records = payload.values() if isinstance(payload, dict) else payload
    rows: list[dict[str, object]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        ticker = record.get("ticker")
        cik = record.get("cik_str")
        title = record.get("title")

        if ticker is None or cik is None:
            continue

        rows.append(
            {
                "ticker": _normalize_ticker(ticker),
                "cik": int(cik),
                "cik_padded": format_cik(cik),
                "title": "" if title is None else str(title).strip(),
            }
        )

    columns = ["ticker", "cik", "cik_padded", "title"]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    return df.sort_values("ticker").reset_index(drop=True)


def fetch_company_tickers(
    url: str = SEC_COMPANY_TICKERS_URL,
    user_agent: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and normalize the SEC ticker-to-CIK mapping."""
    client = session or requests.Session()
    response = client.get(url, headers=_headers(user_agent), timeout=30)
    response.raise_for_status()
    return normalize_company_tickers(response.json())


def lookup_ciks(
    symbols: list[str],
    company_tickers: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Return CIK metadata for requested symbols."""
    if not symbols:
        raise ValueError("symbols cannot be empty.")

    requested = pd.DataFrame(
        {"symbol": [_normalize_ticker(symbol) for symbol in symbols]}
    ).drop_duplicates(subset=["symbol"], keep="first")

    required_columns = {"ticker", "cik", "cik_padded", "title"}
    missing_columns = required_columns - set(company_tickers.columns)
    if missing_columns:
        raise ValueError(
            "company_tickers is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    lookup = company_tickers.copy()
    lookup["ticker"] = lookup["ticker"].map(_normalize_ticker)
    result = requested.merge(
        lookup[["ticker", "cik", "cik_padded", "title"]],
        left_on="symbol",
        right_on="ticker",
        how="left",
    )
    result = result[["symbol", "ticker", "cik", "cik_padded", "title"]]

    if strict and result["cik"].isna().any():
        missing = result.loc[result["cik"].isna(), "symbol"].tolist()
        raise KeyError("Missing SEC CIKs for symbols: " + ", ".join(missing))

    return result


def save_company_tickers(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_ticker_cik_lookup(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
