import pytest

from market_qml.ingestion.sec import (
    format_cik,
    lookup_ciks,
    normalize_company_submissions,
    normalize_company_tickers,
    normalize_submissions,
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


def test_normalize_company_submissions_keeps_target_forms():
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001", "0002", "0003", "0004"],
                "filingDate": ["2024-01-02", "2024-02-03", "2024-03-04", "2024-04-05"],
                "reportDate": ["2023-12-31", "2024-01-31", "", "2024-03-31"],
                "form": ["10-K", "S-1", "8-K", "10-Q"],
                "primaryDocument": ["aapl-10k.htm", "aapl-s1.htm", "aapl-8k.htm", "aapl-10q.htm"],
            }
        }
    }

    df = normalize_company_submissions("aapl", 320193, payload)

    assert df["symbol"].tolist() == ["AAPL", "AAPL", "AAPL"]
    assert set(df["form"]) == {"10-K", "10-Q", "8-K"}
    assert df["cik_padded"].unique().tolist() == ["0000320193"]
    assert df["accession_number"].tolist() == ["0001", "0003", "0004"]
    assert df["primary_document"].tolist() == ["aapl-10k.htm", "aapl-8k.htm", "aapl-10q.htm"]


def test_normalize_submissions_combines_lookup_payloads():
    lookup = normalize_company_tickers(
        [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        ]
    ).rename(columns={"ticker": "symbol"})
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001"],
                "filingDate": ["2024-01-02"],
                "reportDate": ["2023-12-31"],
                "form": ["10-K"],
                "primaryDocument": ["filing.htm"],
            }
        }
    }

    df = normalize_submissions({"AAPL": payload, "MSFT": payload}, lookup)

    assert df["symbol"].tolist() == ["AAPL", "MSFT"]
    assert df["form"].tolist() == ["10-K", "10-K"]
