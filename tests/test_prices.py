import pandas as pd

from market_qml.ingestion.prices import _normalize_bar_pages


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