import pandas as pd
import pytest

import market_qml.ingestion.prices as price_ingestion
from market_qml.ingestion.prices import _normalize_bar_pages, normalize_asset_snapshot
from scripts.ingest_alpaca_prices import load_candidate_symbols, merge_price_history
from scripts.snapshot_alpaca_assets import append_asset_snapshot


def test_normalize_bar_pages():
    pages = [
        {
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
                ],
                "SPY": [
                    {
                        "t": "2024-01-02T05:00:00Z",
                        "o": 470.0,
                        "h": 472.0,
                        "l": 469.0,
                        "c": 471.0,
                        "v": 2000000,
                        "n": 22000,
                        "vw": 470.85,
                    }
                ],
            },
            "next_page_token": None,
        }
    ]

    df = _normalize_bar_pages(pages)

    assert len(df) == 2
    assert set(df["symbol"]) == {"AAPL", "SPY"}
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
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df.loc[df["symbol"] == "AAPL", "close"].iloc[0] == 181.0


def test_asset_snapshots_are_effective_dated_and_append_only(tmp_path):
    first = normalize_asset_snapshot(
        [
            {
                "id": "1",
                "class": "us_equity",
                "exchange": "NASDAQ",
                "symbol": "AAA",
                "status": "active",
                "tradable": True,
            },
            {
                "id": "2",
                "class": "us_equity",
                "exchange": "NYSE",
                "symbol": "OLD",
                "status": "inactive",
                "tradable": False,
            },
        ],
        snapshot_date="2024-01-02",
    )
    second = normalize_asset_snapshot(
        [
            {
                "id": "1",
                "class": "us_equity",
                "exchange": "NASDAQ",
                "symbol": "AAA",
                "status": "inactive",
                "tradable": False,
            }
        ],
        snapshot_date="2024-01-03",
    )
    path = tmp_path / "asset_history.parquet"
    append_asset_snapshot(first, path)
    history = append_asset_snapshot(second, path)

    assert len(history) == 3
    assert set(history["symbol"]) == {"AAA", "OLD"}
    assert history.loc[history["symbol"].eq("AAA"), "status"].tolist() == [
        "active",
        "inactive",
    ]


def test_asset_snapshot_defaults_to_paper_host_and_allows_live_override(monkeypatch):
    calls = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return []

    monkeypatch.setattr(
        price_ingestion.requests,
        "get",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.delenv("ALPACA_TRADING_BASE_URL", raising=False)
    price_ingestion.fetch_alpaca_asset_snapshot(snapshot_date="2024-01-02")
    monkeypatch.setenv("ALPACA_TRADING_BASE_URL", "https://api.alpaca.markets/")
    price_ingestion.fetch_alpaca_asset_snapshot(snapshot_date="2024-01-02")

    assert calls[0][0] == "https://paper-api.alpaca.markets/v2/assets"
    assert calls[1][0] == "https://api.alpaca.markets/v2/assets"


def test_asset_snapshot_retries_temporary_server_failures(monkeypatch):
    calls = []
    sleeps = []

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self.text = "temporary failure"
            self._payload = [] if payload is None else payload

        def json(self):
            return self._payload

    responses = [Response(500), Response(503), Response(200)]
    monkeypatch.setattr(
        price_ingestion.requests,
        "get",
        lambda url, **kwargs: calls.append((url, kwargs)) or responses.pop(0),
    )
    monkeypatch.setattr(price_ingestion.time, "sleep", sleeps.append)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    snapshot = price_ingestion.fetch_alpaca_asset_snapshot(
        snapshot_date="2024-01-02",
        initial_backoff_seconds=0.5,
    )

    assert snapshot.empty
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]


def test_asset_snapshot_fails_immediately_for_authentication_error(monkeypatch):
    calls = []
    sleeps = []

    class Response:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr(
        price_ingestion.requests,
        "get",
        lambda url, **kwargs: calls.append((url, kwargs)) or Response(),
    )
    monkeypatch.setattr(price_ingestion.time, "sleep", sleeps.append)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    with pytest.raises(RuntimeError, match="Confirm the host matches"):
        price_ingestion.fetch_alpaca_asset_snapshot(snapshot_date="2024-01-02")

    assert len(calls) == 1
    assert sleeps == []


def test_asset_snapshot_reports_exhausted_temporary_failures(monkeypatch):
    sleeps = []

    class Response:
        status_code = 500
        text = "internal server error occurred"

    monkeypatch.setattr(
        price_ingestion.requests,
        "get",
        lambda url, **kwargs: Response(),
    )
    monkeypatch.setattr(price_ingestion.time, "sleep", sleeps.append)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        price_ingestion.fetch_alpaca_asset_snapshot(
            snapshot_date="2024-01-02",
            max_attempts=3,
            initial_backoff_seconds=0.25,
        )

    assert sleeps == [0.25, 0.5]


def test_asset_snapshot_retries_network_errors(monkeypatch):
    sleeps = []
    responses = [
        price_ingestion.requests.ConnectionError("connection reset"),
        type(
            "Response",
            (),
            {"status_code": 200, "text": "", "json": staticmethod(list)},
        )(),
    ]

    def request(*args, **kwargs):
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(price_ingestion.requests, "get", request)
    monkeypatch.setattr(price_ingestion.time, "sleep", sleeps.append)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")

    snapshot = price_ingestion.fetch_alpaca_asset_snapshot(
        snapshot_date="2024-01-02",
        initial_backoff_seconds=0.1,
    )

    assert snapshot.empty
    assert sleeps == [0.1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts must be at least 1"),
        (
            {"initial_backoff_seconds": -1},
            "initial_backoff_seconds cannot be negative",
        ),
    ],
)
def test_asset_snapshot_validates_retry_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        price_ingestion.fetch_alpaca_asset_snapshot(**kwargs)


def test_candidate_pool_uses_only_latest_tradable_snapshot(tmp_path):
    assets = pd.DataFrame(
        [
            {
                "symbol": "OLD",
                "effective_date": "2024-01-01",
                "asset_class": "us_equity",
                "exchange": "NYSE",
                "status": "active",
                "tradable": True,
            },
            {
                "symbol": "OLD",
                "effective_date": "2024-01-02",
                "asset_class": "us_equity",
                "exchange": "NYSE",
                "status": "inactive",
                "tradable": False,
            },
            {
                "symbol": "AAA",
                "effective_date": "2024-01-02",
                "asset_class": "us_equity",
                "exchange": "NASDAQ",
                "status": "active",
                "tradable": True,
            },
            {
                "symbol": "OTC",
                "effective_date": "2024-01-02",
                "asset_class": "us_equity",
                "exchange": "OTC",
                "status": "active",
                "tradable": True,
            },
        ]
    )
    path = tmp_path / "assets.parquet"
    assets.to_parquet(path, index=False)

    symbols = load_candidate_symbols(
        path, exchanges=["NYSE", "NASDAQ"], limit=10, benchmark="SPY"
    )
    assert set(symbols) == {"AAA", "SPY"}


def test_price_history_merge_preserves_removed_symbols():
    existing = pd.DataFrame(
        {
            "symbol": ["OLD", "AAA"],
            "timestamp": ["2024-01-01T00:00:00Z"] * 2,
            "close": [5.0, 10.0],
        }
    )
    fresh = pd.DataFrame(
        {"symbol": ["AAA"], "timestamp": ["2024-01-01T00:00:00Z"], "close": [11.0]}
    )

    result = merge_price_history(existing, fresh)
    assert set(result["symbol"]) == {"AAA", "OLD"}
    assert result.loc[result["symbol"].eq("AAA"), "close"].item() == 11.0
