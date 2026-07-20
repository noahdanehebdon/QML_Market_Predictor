import pytest
import requests

from market_qml.ingestion import prices
from market_qml.ingestion import sec
from scripts import pull_macro


class MockResponse:
    def __init__(self, *, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class MockSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_fetch_alpaca_bars_uses_mock_pages_and_normalizes_schema(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    mock_session = MockSession(
        [
            MockResponse(
                json_data={
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2024-01-02T05:00:00Z",
                                "o": 180.0,
                                "h": 182.0,
                                "l": 179.0,
                                "c": 181.0,
                                "v": 1000000,
                                "n": 12000,
                                "vw": 180.75,
                            }
                        ]
                    },
                    "next_page_token": "next",
                }
            ),
            MockResponse(
                json_data={
                    "bars": {
                        "MSFT": [
                            {
                                "t": "2024-01-03T05:00:00Z",
                                "o": 370.0,
                                "h": 372.0,
                                "l": 369.0,
                                "c": 371.0,
                                "v": 2000000,
                                "n": 22000,
                                "vw": 370.85,
                            }
                        ]
                    },
                    "next_page_token": None,
                }
            ),
        ]
    )
    monkeypatch.setattr(prices.requests, "Session", lambda: mock_session)

    df, pages = prices.fetch_alpaca_bars(
        prices.PriceRequest(symbols=["AAPL", "MSFT"], start="2024-01-01")
    )

    assert len(pages) == 2
    assert list(df.columns) == [
        "symbol",
        "timestamp",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    ]
    assert df["symbol"].tolist() == ["AAPL", "MSFT"]
    assert mock_session.calls[0]["headers"]["APCA-API-KEY-ID"] == "test-key"
    assert mock_session.calls[1]["params"]["page_token"] == "next"


def test_missing_alpaca_credentials_have_useful_error(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing Alpaca credentials"):
        prices._credentials()


def test_fetch_bls_chunk_uses_mock_response_and_normalizes_schema(monkeypatch):
    def mock_post(url, json, timeout):
        assert url == pull_macro.BLS_URL
        assert json["seriesid"] == ["CUSR0000SA0"]
        assert json["registrationkey"] == "fake-key"
        assert timeout == 30
        return MockResponse(
            json_data={
                "status": "REQUEST_SUCCEEDED",
                "Results": {
                    "series": [
                        {
                            "seriesID": "CUSR0000SA0",
                            "data": [
                                {"year": "2024", "period": "M01", "value": "309.685"},
                                {"year": "2024", "period": "M13", "value": "310.000"},
                            ],
                        }
                    ]
                },
            }
        )

    monkeypatch.setattr(pull_macro.requests, "post", mock_post)

    df = pull_macro.fetch_bls_chunk(
        series_map={"cpi_all_items_sa": "CUSR0000SA0"},
        start_year=2024,
        end_year=2024,
        api_key="fake-key",
    )

    assert list(df.columns) == [
        "date",
        "series_id",
        "column",
        "value",
        "source",
        "retrieved_at",
    ]
    assert len(df) == 1
    assert df.loc[0, "column"] == "cpi_all_items_sa"
    assert df.loc[0, "value"] == 309.685


def test_fetch_fed_ddp_series_uses_mock_csv_and_normalizes_schema(monkeypatch):
    csv_text = "\n".join(
        [
            "Metadata row",
            "Time Period,RIFLGFCY10_N.B",
            "2024-01-02,4.01",
            "2024-01-03,ND",
            "1999-12-31,6.50",
        ]
    )

    def mock_get(url, timeout):
        assert url == "https://example.test/fed.csv"
        assert timeout == 30
        return MockResponse(text=csv_text)

    monkeypatch.setattr(pull_macro.requests, "get", mock_get)

    df = pull_macro.fetch_fed_ddp_series(
        column="treasury_10y",
        series_id="RIFLGFCY10_N.B",
        url="https://example.test/fed.csv",
        source="federal_reserve_h15",
        start_year=2024,
    )

    assert list(df.columns) == [
        "date",
        "series_id",
        "column",
        "value",
        "source",
        "retrieved_at",
    ]
    assert len(df) == 1
    assert df.loc[0, "column"] == "treasury_10y"
    assert df.loc[0, "value"] == 4.01


def test_sec_fetchers_use_mock_responses_and_user_agent():
    company_tickers_payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
    }
    submissions_payload = {
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
    company_facts_payload = {
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "fy": 2024,
                                "fp": "FY",
                                "filed": "2024-10-31",
                                "form": "10-K",
                                "val": 100,
                                "end": "2024-09-28",
                            }
                        ]
                    }
                }
            }
        }
    }
    mock_session = MockSession(
        [
            MockResponse(json_data=company_tickers_payload),
            MockResponse(json_data=submissions_payload),
            MockResponse(json_data=company_facts_payload),
        ]
    )

    tickers = sec.fetch_company_tickers(
        user_agent="QML Market Predictor test@example.com",
        session=mock_session,
    )
    submissions = sec.fetch_company_submission(
        320193,
        user_agent="QML Market Predictor test@example.com",
        session=mock_session,
    )
    company_facts = sec.fetch_company_facts(
        320193,
        user_agent="QML Market Predictor test@example.com",
        session=mock_session,
    )

    assert tickers["ticker"].tolist() == ["AAPL"]
    assert submissions["filings"]["recent"]["form"] == ["10-K"]
    assert company_facts["facts"]["us-gaap"]["Assets"]["units"]["USD"][0]["val"] == 100
    assert all(
        call["headers"]["User-Agent"] == "QML Market Predictor test@example.com"
        for call in mock_session.calls
    )
    assert mock_session.calls[1]["url"].endswith("/CIK0000320193.json")
    assert mock_session.calls[2]["url"].endswith("/CIK0000320193.json")


def test_missing_sec_user_agent_has_useful_error(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.delenv("USER_AGENT", raising=False)

    with pytest.raises(RuntimeError, match="Missing SEC user agent"):
        sec._headers()


def test_sec_session_retries_transient_get_failures():
    session = sec.build_sec_session()
    adapter = session.get_adapter("https://")

    assert isinstance(session, requests.Session)
    assert adapter.max_retries.total == 3
    assert 429 in adapter.max_retries.status_forcelist
    assert adapter.max_retries.respect_retry_after_header is True


def test_sec_request_pacing_enforces_five_per_second(monkeypatch):
    clock = iter([10.0, 10.05])
    sleeps = []
    monkeypatch.setattr(sec.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(sec.time, "sleep", sleeps.append)

    first = sec.pace_sec_requests(None)
    second = sec.pace_sec_requests(first)

    assert first == 10.0
    assert second == pytest.approx(10.2)
    assert sleeps == pytest.approx([0.15])
