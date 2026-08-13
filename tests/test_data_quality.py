from __future__ import annotations

import pandas as pd

from market_qml.ingestion.data_quality import validate_snapshot

NOW = pd.Timestamp("2026-01-10", tz="UTC")


def _valid_inputs():
    provenance = pd.Timestamp("2026-01-09", tz="UTC")
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "timestamp": pd.to_datetime(["2026-01-07", "2026-01-08"], utc=True),
            "date": pd.to_datetime(["2026-01-07", "2026-01-08"]).date,
            "open": [10.0, 10.5],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000, 1200],
            "ingested_at_utc": [provenance, provenance],
        }
    )
    assets = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "effective_date": [pd.Timestamp("2026-01-08")],
            "security_type": ["common_stock"],
            "status": ["active"],
            "tradable": [True],
            "ingested_at_utc": [provenance],
        }
    )
    submissions = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "filing_date": [pd.Timestamp("2026-01-08")],
            "report_date": [pd.Timestamp("2025-12-31")],
            "accession_number": ["1"],
            "ingested_at_utc": [provenance],
            "earliest_tradable_date": [pd.Timestamp("2026-01-09")],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "filing_date": [pd.Timestamp("2026-01-08")],
            "end_date": [pd.Timestamp("2025-12-31")],
            "accession_number": ["1"],
            "concept": ["revenue"],
            "unit": ["USD"],
            "value": [100.0],
            "ingested_at_utc": [provenance],
            "earliest_tradable_date": [pd.Timestamp("2026-01-09")],
        }
    )
    macro = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-12-01")],
            "series_id": ["CPI"],
            "value": [100.0],
            "retrieved_at": [provenance],
        }
    )
    return prices, assets, submissions, fundamentals, macro


def test_valid_snapshot_passes_contracts():
    prices, assets, submissions, fundamentals, macro = _valid_inputs()
    result = validate_snapshot(
        prices=prices,
        assets=assets,
        submissions=submissions,
        fundamentals=fundamentals,
        macro_raw=macro,
        now=NOW,
    )

    assert result.passed
    assert result.checks.loc[result.checks.severity.eq("critical"), "passed"].all()


def test_invalid_ohlc_and_sec_availability_fail_closed():
    prices, assets, submissions, fundamentals, macro = _valid_inputs()
    prices.loc[0, "high"] = 8.0
    submissions.loc[0, "filing_date"] = pd.Timestamp("2025-01-01")
    result = validate_snapshot(
        prices=prices,
        assets=assets,
        submissions=submissions,
        fundamentals=fundamentals,
        macro_raw=macro,
        now=NOW,
    )

    assert not result.passed
    failed = set(result.checks.loc[~result.checks.passed, "check"])
    assert "prices.valid_ohlcv" in failed
    assert "submissions.availability_order" in failed


def test_extreme_adjusted_return_is_quarantined_but_not_silently_clipped():
    prices, assets, submissions, fundamentals, macro = _valid_inputs()
    prices.loc[1, ["open", "high", "low", "close"]] = [25.0, 25.0, 25.0, 25.0]
    result = validate_snapshot(
        prices=prices,
        assets=assets,
        submissions=submissions,
        fundamentals=fundamentals,
        macro_raw=macro,
        now=NOW,
    )

    assert result.passed
    assert result.quarantine["reason"].tolist() == ["absolute_daily_return_over_100pct"]
