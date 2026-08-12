"""Configuration parsing for macroeconomic data sources."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_series_config(
    config_path: Path, expected_columns: list[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Macro data source config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    macro = config.get("macro")
    if not isinstance(macro, dict):
        raise ValueError(f"Missing 'macro' section in {config_path}")

    bls_series = {
        item["column"]: item["series_id"]
        for item in macro.get("bls_api", {}).values()
        if item.get("column") and item.get("series_id")
    }
    fed_series = {}
    for release, series in macro.get("federal_reserve_ddp", {}).items():
        for item in series.values():
            if item.get("column") and item.get("series_id") and item.get("url"):
                fed_series[item["column"]] = {
                    "series_id": item["series_id"],
                    "url": item["url"],
                    "source": item.get("source", f"federal_reserve_{release}"),
                }

    missing = [
        column
        for column in expected_columns
        if column not in bls_series and column not in fed_series
    ]
    if missing:
        raise ValueError(
            "Macro config is missing expected columns: " + ", ".join(missing)
        )
    if not bls_series:
        raise ValueError(f"No BLS macro series configured in {config_path}")
    if not fed_series:
        raise ValueError(
            f"No Federal Reserve DDP macro series configured in {config_path}"
        )
    return bls_series, fed_series
