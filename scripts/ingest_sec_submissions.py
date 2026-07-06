"""Fetch and normalize SEC submissions for the configured universe."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from market_qml.ingestion.sec import (
    fetch_company_submission,
    normalize_submissions,
    save_raw_submission,
    save_submissions,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_SOURCES_CONFIG_PATH = Path("configs/data_sources.yaml")
DEFAULT_LOOKUP_PATH = Path("data/processed/sec_ticker_cik_lookup.parquet")
DEFAULT_RAW_DIR = Path("data/raw/sec")
DEFAULT_OUTPUT_PATH = Path("data/processed/sec_submissions.parquet")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sec_submissions_url_template(
    config_path: Path = DEFAULT_DATA_SOURCES_CONFIG_PATH,
) -> str:
    config = _load_yaml(config_path)
    sec = config.get("sec")

    if not isinstance(sec, dict):
        raise ValueError(f"Missing 'sec' section in {config_path}")

    url_template = sec.get("submissions_url_template")
    if not url_template:
        raise ValueError(f"Missing sec.submissions_url_template in {config_path}")

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
        description="Fetch SEC submissions metadata for the configured universe."
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
        help="Directory for raw SEC submissions JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for normalized SEC submissions.",
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

    url_template = load_sec_submissions_url_template(args.data_sources_config)
    lookup = load_ticker_cik_lookup(args.lookup_path)
    lookup = _filter_symbols(lookup, args.symbols)

    submissions: dict[str, dict] = {}
    for row in lookup.itertuples(index=False):
        symbol = str(row.symbol)
        LOGGER.info("Fetching SEC submissions for %s.", symbol)
        payload = fetch_company_submission(
            cik=row.cik_padded,
            url_template=url_template,
        )
        save_raw_submission(payload, args.raw_dir / f"{symbol}_submissions.json")
        submissions[symbol] = payload

    normalized = normalize_submissions(submissions=submissions, ticker_cik_lookup=lookup)
    save_submissions(normalized, args.output)

    LOGGER.info("Saved raw SEC submissions to: %s", args.raw_dir)
    LOGGER.info("Saved normalized SEC submissions to: %s", args.output)
    LOGGER.info("Rows: %s", len(normalized))
    LOGGER.info("Forms: %s", ", ".join(sorted(normalized["form"].dropna().unique())))


if __name__ == "__main__":
    main()
