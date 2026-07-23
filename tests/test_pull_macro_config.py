from pathlib import Path

from scripts.pull_macro import EXPECTED_COLUMNS, load_macro_config


def test_load_macro_config_reads_expected_series():
    bls_series, fed_series = load_macro_config(Path("configs/data_sources.yaml"))

    configured_columns = set(bls_series) | set(fed_series)

    assert set(EXPECTED_COLUMNS) <= configured_columns
    assert bls_series["cpi_all_items_sa"] == "CUSR0000SA0"
    assert bls_series["unemployment_rate"] == "LNS14000000"
    assert fed_series["treasury_10y"]["series_id"] == "RIFLGFCY10_N.B"
    assert fed_series["treasury_10y"]["url"].startswith(
        "https://www.federalreserve.gov/"
    )
