import pandas as pd

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
            {"id": "1", "class": "us_equity", "exchange": "NASDAQ", "symbol": "AAA", "status": "active", "tradable": True},
            {"id": "2", "class": "us_equity", "exchange": "NYSE", "symbol": "OLD", "status": "inactive", "tradable": False},
        ],
        snapshot_date="2024-01-02",
    )
    second = normalize_asset_snapshot(
        [{"id": "1", "class": "us_equity", "exchange": "NASDAQ", "symbol": "AAA", "status": "inactive", "tradable": False}],
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


def test_candidate_pool_uses_only_latest_tradable_snapshot(tmp_path):
    assets = pd.DataFrame(
        [
            {"symbol": "OLD", "effective_date": "2024-01-01", "asset_class": "us_equity", "exchange": "NYSE", "status": "active", "tradable": True},
            {"symbol": "OLD", "effective_date": "2024-01-02", "asset_class": "us_equity", "exchange": "NYSE", "status": "inactive", "tradable": False},
            {"symbol": "AAA", "effective_date": "2024-01-02", "asset_class": "us_equity", "exchange": "NASDAQ", "status": "active", "tradable": True},
            {"symbol": "OTC", "effective_date": "2024-01-02", "asset_class": "us_equity", "exchange": "OTC", "status": "active", "tradable": True},
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
        {"symbol": ["OLD", "AAA"], "timestamp": ["2024-01-01T00:00:00Z"] * 2, "close": [5.0, 10.0]}
    )
    fresh = pd.DataFrame(
        {"symbol": ["AAA"], "timestamp": ["2024-01-01T00:00:00Z"], "close": [11.0]}
    )

    result = merge_price_history(existing, fresh)
    assert set(result["symbol"]) == {"AAA", "OLD"}
    assert result.loc[result["symbol"].eq("AAA"), "close"].item() == 11.0
