"""SEC ingestion helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from market_qml.ingestion.sec_io import save_json, save_parquet

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_FACTS_URL_TEMPLATE = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)
SEC_SUBMISSION_FORMS = {"10-K", "10-Q", "8-K"}
SEC_REQUEST_INTERVAL_SECONDS = 0.2
SEC_FUNDAMENTAL_CONCEPTS = {
    "revenue": [
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "debt": ["LongTermDebt", "LongTermDebtAndFinanceLeaseObligations"],
    "gross_profit": ["GrossProfit"],
    "interest_expense": ["InterestExpenseNonOperating", "InterestExpense"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "stock_based_compensation": ["ShareBasedCompensation"],
    "shares_outstanding": ["EntityCommonStockSharesOutstanding"],
}


def build_sec_session() -> requests.Session:
    """Build a session that retries temporary SEC and network failures."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def pace_sec_requests(
    last_request_at: float | None,
    *,
    interval_seconds: float = SEC_REQUEST_INTERVAL_SECONDS,
) -> float:
    """Wait as needed to keep SEC request starts at or below five per second."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    now = time.monotonic()
    if last_request_at is not None:
        wait_seconds = max(0.0, interval_seconds - (now - last_request_at))
        if wait_seconds:
            time.sleep(wait_seconds)
            now += wait_seconds
    return now


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


def normalize_company_tickers(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
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


def fetch_company_submission(
    cik: object,
    url_template: str = SEC_SUBMISSIONS_URL_TEMPLATE,
    user_agent: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch the SEC submissions JSON for one company CIK."""
    cik_padded = format_cik(cik)
    client = session or requests.Session()
    response = client.get(
        url_template.format(cik=cik_padded),
        headers=_headers(user_agent),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def fetch_company_facts(
    cik: object,
    url_template: str = SEC_COMPANY_FACTS_URL_TEMPLATE,
    user_agent: str | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch the SEC companyfacts JSON for one company CIK."""
    cik_padded = format_cik(cik)
    client = session or requests.Session()
    response = client.get(
        url_template.format(cik=cik_padded),
        headers=_headers(user_agent),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _recent_filings(payload: dict[str, Any]) -> dict[str, list[Any]]:
    filings = payload.get("filings", {})
    if not isinstance(filings, dict):
        return {}

    recent = filings.get("recent", {})
    if not isinstance(recent, dict):
        return {}

    return {key: value for key, value in recent.items() if isinstance(value, list)}


def normalize_company_submissions(
    symbol: str,
    cik: object,
    payload: dict[str, Any],
    *,
    forms: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize one company's recent SEC submissions into filing rows."""
    allowed_forms = forms or SEC_SUBMISSION_FORMS
    recent = _recent_filings(payload)

    columns = [
        "symbol",
        "cik",
        "cik_padded",
        "form",
        "filing_date",
        "report_date",
        "accession_number",
        "primary_document",
    ]

    if not recent:
        return pd.DataFrame(columns=columns)

    row_count = max((len(values) for values in recent.values()), default=0)
    rows: list[dict[str, object]] = []
    cik_padded = format_cik(cik)
    normalized_symbol = _normalize_ticker(symbol)

    for index in range(row_count):
        form = _value_at(recent, "form", index)
        if form not in allowed_forms:
            continue

        rows.append(
            {
                "symbol": normalized_symbol,
                "cik": int(cik),
                "cik_padded": cik_padded,
                "form": form,
                "filing_date": _value_at(recent, "filingDate", index),
                "report_date": _value_at(recent, "reportDate", index),
                "accession_number": _value_at(recent, "accessionNumber", index),
                "primary_document": _value_at(recent, "primaryDocument", index),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    return df.sort_values(["symbol", "filing_date", "accession_number"]).reset_index(
        drop=True
    )


def _value_at(values_by_name: dict[str, list[Any]], name: str, index: int) -> Any:
    values = values_by_name.get(name, [])
    if index >= len(values):
        return None

    return values[index]


def normalize_submissions(
    submissions: dict[str, dict[str, Any]],
    ticker_cik_lookup: pd.DataFrame,
    *,
    forms: set[str] | None = None,
) -> pd.DataFrame:
    """Normalize submissions payloads for all symbols in a lookup table."""
    frames: list[pd.DataFrame] = []

    for row in ticker_cik_lookup.itertuples(index=False):
        symbol = str(row.symbol)
        payload = submissions.get(symbol)
        if payload is None:
            continue

        frames.append(
            normalize_company_submissions(
                symbol=symbol,
                cik=getattr(row, "cik_padded", row.cik),
                payload=payload,
                forms=forms,
            )
        )

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "cik",
                "cik_padded",
                "form",
                "filing_date",
                "report_date",
                "accession_number",
                "primary_document",
            ]
        )

    return pd.concat(frames, ignore_index=True)


def normalize_company_facts(
    symbol: str,
    cik: object,
    payload: dict[str, Any],
    *,
    concept_map: dict[str, list[str]] | None = None,
    taxonomy: str = "us-gaap",
) -> pd.DataFrame:
    """Normalize selected SEC companyfacts concepts into long-format rows."""
    concepts = concept_map or SEC_FUNDAMENTAL_CONCEPTS
    facts = payload.get("facts", {})
    taxonomy_facts = facts.get(taxonomy, {}) if isinstance(facts, dict) else {}

    columns = [
        "symbol",
        "ticker",
        "cik",
        "cik_padded",
        "fiscal_year",
        "fiscal_period",
        "filing_date",
        "form",
        "concept",
        "taxonomy",
        "sec_concept",
        "value",
        "unit",
        "start_date",
        "end_date",
        "accession_number",
        "frame",
    ]

    if not isinstance(taxonomy_facts, dict):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    cik_padded = format_cik(cik)
    normalized_symbol = _normalize_ticker(symbol)

    for concept, sec_concepts in concepts.items():
        for sec_concept in sec_concepts:
            fact = taxonomy_facts.get(sec_concept)
            if not isinstance(fact, dict):
                continue

            units = fact.get("units", {})
            if not isinstance(units, dict):
                continue

            for unit, observations in units.items():
                if not isinstance(observations, list):
                    continue

                for observation in observations:
                    if not isinstance(observation, dict):
                        continue

                    value = observation.get("val")
                    if value is None:
                        continue

                    rows.append(
                        {
                            "symbol": normalized_symbol,
                            "ticker": normalized_symbol,
                            "cik": int(cik),
                            "cik_padded": cik_padded,
                            "fiscal_year": observation.get("fy"),
                            "fiscal_period": observation.get("fp"),
                            "filing_date": observation.get("filed"),
                            "form": observation.get("form"),
                            "concept": concept,
                            "taxonomy": taxonomy,
                            "sec_concept": sec_concept,
                            "value": value,
                            "unit": unit,
                            "start_date": observation.get("start"),
                            "end_date": observation.get("end"),
                            "accession_number": observation.get("accn"),
                            "frame": observation.get("frame"),
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df.sort_values(
        ["symbol", "concept", "filing_date", "fiscal_year", "fiscal_period"]
    ).reset_index(drop=True)


def normalize_fundamentals(
    company_facts: dict[str, dict[str, Any]],
    ticker_cik_lookup: pd.DataFrame,
    *,
    concept_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Normalize companyfacts payloads for all symbols in a lookup table."""
    frames: list[pd.DataFrame] = []

    for row in ticker_cik_lookup.itertuples(index=False):
        symbol = str(row.symbol)
        payload = company_facts.get(symbol)
        if payload is None:
            continue

        frames.append(
            normalize_company_facts(
                symbol=symbol,
                cik=getattr(row, "cik_padded", row.cik),
                payload=payload,
                concept_map=concept_map,
            )
        )

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "ticker",
                "cik",
                "cik_padded",
                "fiscal_year",
                "fiscal_period",
                "filing_date",
                "form",
                "concept",
                "taxonomy",
                "sec_concept",
                "value",
                "unit",
                "start_date",
                "end_date",
                "accession_number",
                "frame",
            ]
        )

    return pd.concat(frames, ignore_index=True)


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
    save_parquet(df, output_path)


def save_ticker_cik_lookup(df: pd.DataFrame, output_path: str | Path) -> None:
    save_parquet(df, output_path)


def save_raw_submission(payload: dict[str, Any], output_path: str | Path) -> None:
    save_json(payload, output_path)


def save_submissions(df: pd.DataFrame, output_path: str | Path) -> None:
    save_parquet(df, output_path)


def save_raw_company_facts(payload: dict[str, Any], output_path: str | Path) -> None:
    save_json(payload, output_path)


def save_fundamentals(df: pd.DataFrame, output_path: str | Path) -> None:
    save_parquet(df, output_path)
