"""Fetch and normalize SEC companyfacts fundamentals for the configured universe."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from market_qml.ingestion.sec import (
    build_sec_session,
    fetch_company_facts,
    normalize_fundamentals,
    pace_sec_requests,
    save_fundamentals,
    save_raw_company_facts,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_SOURCES_CONFIG_PATH = Path("configs/data_sources.yaml")
DEFAULT_LOOKUP_PATH = Path("data/processed/sec_ticker_cik_lookup.parquet")
DEFAULT_RAW_DIR = Path("data/raw/sec")
DEFAULT_OUTPUT_PATH = Path("data/processed/fundamentals.parquet")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sec_company_facts_url_template(
    config_path: Path = DEFAULT_DATA_SOURCES_CONFIG_PATH,
) -> str:
    config = _load_yaml(config_path)
    sec = config.get("sec")

    if not isinstance(sec, dict):
        raise ValueError(f"Missing 'sec' section in {config_path}")

    url_template = sec.get("company_facts_url_template")
    if not url_template:
        raise ValueError(f"Missing sec.company_facts_url_template in {config_path}")

    return str(url_template)


def load_ticker_cik_lookup(path: Path = DEFAULT_LOOKUP_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Ticker-to-CIK lookup not found: {path}. "
            "Run python -m scripts.build_sec_ticker_cik_lookup first."
        )

    lookup = pd.read_parquet(path)
    required_columns = {"symbol", "cik", "cik_padded"}
    missing_columns = required_columns - set(lookup.columns)
    if missing_columns:
        raise ValueError(
            "Ticker-to-CIK lookup is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    return lookup


def _filter_symbols(lookup: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if not symbols:
        return lookup

    requested = {symbol.strip().upper() for symbol in symbols}
    filtered = lookup[lookup["symbol"].str.upper().isin(requested)].copy()
    found = set(filtered["symbol"].str.upper())
    missing = sorted(requested - found)

    if missing:
        raise KeyError("Missing symbols in ticker-to-CIK lookup: " + ", ".join(missing))

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC companyfacts fundamentals for the configured universe."
    )
    parser.add_argument(
        "--data-sources-config",
        type=Path,
        default=DEFAULT_DATA_SOURCES_CONFIG_PATH,
        help="Path to data-source YAML config.",
    )
    parser.add_argument(
        "--lookup-path",
        type=Path,
        default=DEFAULT_LOOKUP_PATH,
        help="Path to the ticker-to-CIK lookup parquet.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw SEC companyfacts JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for normalized fundamentals.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional subset of symbols to fetch, such as AAPL MSFT.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    args = parse_args()

    url_template = load_sec_company_facts_url_template(args.data_sources_config)
    lookup = load_ticker_cik_lookup(args.lookup_path)
    lookup = _filter_symbols(lookup, args.symbols)

    company_facts: dict[str, dict] = {}
    session = build_sec_session()
    last_request_at: float | None = None
    for row in lookup.itertuples(index=False):
        symbol = str(row.symbol)
        LOGGER.info("Fetching SEC companyfacts for %s.", symbol)
        last_request_at = pace_sec_requests(last_request_at)
        try:
            payload = fetch_company_facts(
                cik=row.cik_padded,
                url_template=url_template,
                session=session,
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                LOGGER.warning(
                    "Skipping SEC companyfacts for %s; endpoint returned 404.",
                    symbol,
                )
                continue
            raise

        save_raw_company_facts(payload, args.raw_dir / f"{symbol}_companyfacts.json")
        company_facts[symbol] = payload

    normalized = normalize_fundamentals(
        company_facts=company_facts,
        ticker_cik_lookup=lookup,
    )
    save_fundamentals(normalized, args.output)

    LOGGER.info("Saved raw SEC companyfacts to: %s", args.raw_dir)
    LOGGER.info("Saved normalized fundamentals to: %s", args.output)
    LOGGER.info("Rows: %s", len(normalized))
    LOGGER.info(
        "Concepts: %s", ", ".join(sorted(normalized["concept"].dropna().unique()))
    )


if __name__ == "__main__":
    main()
