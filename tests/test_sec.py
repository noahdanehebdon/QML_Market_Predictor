import pytest

from market_qml.ingestion.sec import (
    format_cik,
    lookup_ciks,
    normalize_company_tickers,
)


def test_normalize_company_tickers_pads_cik_and_uppercases_ticker():
    payload = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }

    df = normalize_company_tickers(payload)

    assert list(df.columns) == ["ticker", "cik", "cik_padded", "title"]
    assert df.loc[df["ticker"] == "AAPL", "cik"].iloc[0] == 320193
    assert df.loc[df["ticker"] == "AAPL", "cik_padded"].iloc[0] == "0000320193"
    assert df.loc[df["ticker"] == "MSFT", "cik_padded"].iloc[0] == "0000789019"


def test_lookup_ciks_is_case_insensitive_and_preserves_request_order():
    company_tickers = normalize_company_tickers(
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        ]
    )

    result = lookup_ciks(["msft", "AAPL", "MSFT"], company_tickers)

    assert result["symbol"].tolist() == ["MSFT", "AAPL"]
    assert result["cik_padded"].tolist() == ["0000789019", "0000320193"]


def test_lookup_ciks_raises_for_missing_symbol_by_default():
    company_tickers = normalize_company_tickers(
        [{"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}]
    )

    with pytest.raises(KeyError, match="Missing SEC CIKs for symbols: MSFT"):
        lookup_ciks(["AAPL", "MSFT"], company_tickers)


def test_format_cik_rejects_invalid_values():
    assert format_cik("123") == "0000000123"

    with pytest.raises(ValueError, match="Invalid CIK value"):
        format_cik("not-a-cik")
