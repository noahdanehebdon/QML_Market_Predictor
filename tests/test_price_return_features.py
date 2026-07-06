import pandas as pd
import pytest

from market_qml.features.returns import (
    RETURN_WINDOWS,
    add_return_features,
    build_price_return_features,
)


def _price_rows(symbol: str, closes: list[float]) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i),
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": 1000 + i,
            "trade_count": 100 + i,
            "vwap": close - 0.25,
        }
        for i, close in enumerate(closes)
    ]


def test_add_return_features_preserves_price_columns_and_adds_returns():
    prices = pd.DataFrame(_price_rows("AAPL", [100, 110, 121]))

    result = add_return_features(prices, windows=[1])

    assert list(result.columns) == [
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
        "return_1d",
    ]
    assert pd.isna(result.loc[0, "return_1d"])
    assert result.loc[1, "return_1d"] == pytest.approx(0.10)
    assert result.loc[2, "return_1d"] == pytest.approx(0.10)


def test_add_return_features_computes_returns_by_symbol_without_leakage():
    prices = pd.DataFrame(
        _price_rows("AAPL", [100, 110, 121])
        + _price_rows("MSFT", [200, 180, 162])
    )
    prices = prices.sort_values(["date", "symbol"], ascending=[True, False])

    result = add_return_features(prices, windows=[1])

    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    msft = result[result["symbol"] == "MSFT"].reset_index(drop=True)

    assert pd.isna(aapl.loc[0, "return_1d"])
    assert pd.isna(msft.loc[0, "return_1d"])
    assert aapl.loc[1, "return_1d"] == pytest.approx(0.10)
    assert msft.loc[1, "return_1d"] == pytest.approx(-0.10)


def test_add_return_features_uses_past_prices_for_multi_day_windows():
    prices = pd.DataFrame(_price_rows("AAPL", [100, 110, 120, 130, 140, 150]))

    result = add_return_features(prices, windows=[5])

    assert result["return_5d"].iloc[:5].isna().all()
    assert result.loc[5, "return_5d"] == pytest.approx(0.50)


def test_add_return_features_adds_default_windows():
    prices = pd.DataFrame(_price_rows("AAPL", list(range(100, 170))))

    result = add_return_features(prices)

    for window in RETURN_WINDOWS:
        assert f"return_{window}d" in result.columns

    assert result.loc[60, "return_60d"] == pytest.approx(0.60)


def test_add_return_features_requires_price_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        add_return_features(pd.DataFrame({"symbol": ["AAPL"], "close": [100]}))


def test_build_price_return_features_saves_output(tmp_path):
    price_path = tmp_path / "prices.parquet"
    output_path = tmp_path / "price_return_features.parquet"
    prices = pd.DataFrame(_price_rows("AAPL", [100, 110]))
    prices.to_parquet(price_path, index=False)

    result = build_price_return_features(
        price_path=price_path,
        output_path=output_path,
        windows=[1],
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert saved.loc[1, "return_1d"] == pytest.approx(0.10)
