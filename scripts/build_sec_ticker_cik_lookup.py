"""Build a SEC ticker-to-CIK lookup for the configured universe."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from market_qml.ingestion.sec import (
    fetch_company_tickers,
    lookup_ciks,
    save_company_tickers,
    save_ticker_cik_lookup,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")
DEFAULT_DATA_SOURCES_CONFIG_PATH = Path("configs/data_sources.yaml")
DEFAULT_RAW_OUTPUT_PATH = Path("data/raw/sec_company_tickers.parquet")
DEFAULT_PROCESSED_OUTPUT_PATH = Path("data/processed/sec_ticker_cik_lookup.parquet")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_universe_symbols(config_path: Path = DEFAULT_UNIVERSE_CONFIG_PATH) -> list[str]:
    config = _load_yaml(config_path)
    universe = config.get("universe")

    if not isinstance(universe, dict):
        raise ValueError(f"Missing 'universe' section in {config_path}")

    symbols = list(universe.get("symbols") or [])
    benchmark = universe.get("benchmark")

    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)

    if not symbols:
        raise ValueError(f"No universe symbols configured in {config_path}")

    return symbols


def load_sec_company_tickers_url(
    config_path: Path = DEFAULT_DATA_SOURCES_CONFIG_PATH,
) -> str:
    config = _load_yaml(config_path)
    sec = config.get("sec")

    if not isinstance(sec, dict):
        raise ValueError(f"Missing 'sec' section in {config_path}")

    url = sec.get("company_tickers_url")
    if not url:
        raise ValueError(f"Missing sec.company_tickers_url in {config_path}")

    return str(url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SEC ticker-to-CIK lookup files for the configured universe."
    )
    parser.add_argument(
        "--universe-config",
        type=Path,
        default=DEFAULT_UNIVERSE_CONFIG_PATH,
        help="Path to universe YAML config.",
    )
    parser.add_argument(
        "--data-sources-config",
        type=Path,
        default=DEFAULT_DATA_SOURCES_CONFIG_PATH,
        help="Path to data-source YAML config.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_PATH,
        help="Output path for the full SEC company ticker table.",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=DEFAULT_PROCESSED_OUTPUT_PATH,
        help="Output path for the universe ticker-to-CIK lookup.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()
    args = parse_args()

    symbols = load_universe_symbols(args.universe_config)
    url = load_sec_company_tickers_url(args.data_sources_config)

    LOGGER.info("Fetching SEC company ticker metadata.")
    company_tickers = fetch_company_tickers(url=url)
    lookup = lookup_ciks(symbols=symbols, company_tickers=company_tickers)

    save_company_tickers(company_tickers, args.raw_output)
    save_ticker_cik_lookup(lookup, args.processed_output)

    LOGGER.info("Saved full SEC company ticker table to: %s", args.raw_output)
    LOGGER.info("Saved universe ticker-to-CIK lookup to: %s", args.processed_output)
    LOGGER.info("Rows: %s", len(lookup))
    LOGGER.info("Symbols: %s", ", ".join(lookup["symbol"].tolist()))


if __name__ == "__main__":
    main()
