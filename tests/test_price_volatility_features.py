import pandas as pd
import pytest

from market_qml.features.volatility import (
    VOLATILITY_WINDOWS,
    add_volatility_features,
    build_price_volatility_features,
)


def _return_rows(symbol: str, returns: list[float]) -> list[dict]:
    closes = [100.0]
    for daily_return in returns[1:]:
        closes.append(closes[-1] * (1 + daily_return))

    return [
        {
            "symbol": symbol,
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i),
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "open": closes[i] - 1,
            "high": closes[i] + 1,
            "low": closes[i] - 2,
            "close": closes[i],
            "volume": 1000 + i,
            "trade_count": 100 + i,
            "vwap": closes[i] - 0.25,
            "return_1d": returns[i],
            "return_5d": pd.NA,
        }
        for i in range(len(returns))
    ]


def test_add_volatility_features_preserves_existing_columns_and_adds_volatility():
    features = pd.DataFrame(_return_rows("AAPL", [pd.NA, 0.01, 0.02, 0.03]))

    result = add_volatility_features(features, windows=[3], annualization_factor=1)

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
        "return_5d",
        "realized_vol_3d",
    ]
    assert result["realized_vol_3d"].iloc[:3].isna().all()
    assert result.loc[3, "realized_vol_3d"] == pytest.approx(0.0081649658)


def test_add_volatility_features_computes_by_symbol_without_leakage():
    features = pd.DataFrame(
        _return_rows("AAPL", [pd.NA, 0.01, 0.02, 0.03])
        + _return_rows("MSFT", [pd.NA, -0.01, -0.02, -0.03])
    )
    features = features.sort_values(["date", "symbol"], ascending=[True, False])

    result = add_volatility_features(features, windows=[3], annualization_factor=1)
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    msft = result[result["symbol"] == "MSFT"].reset_index(drop=True)

    assert aapl["realized_vol_3d"].iloc[:3].isna().all()
    assert msft["realized_vol_3d"].iloc[:3].isna().all()
    assert aapl.loc[3, "realized_vol_3d"] == pytest.approx(0.0081649658)
    assert msft.loc[3, "realized_vol_3d"] == pytest.approx(0.0081649658)


def test_add_volatility_features_annualizes_rolling_std():
    features = pd.DataFrame(_return_rows("AAPL", [pd.NA, 0.01, 0.02, 0.03]))

    result = add_volatility_features(features, windows=[3], annualization_factor=4)

    assert result.loc[3, "realized_vol_3d"] == pytest.approx(0.0163299316)


def test_add_volatility_features_adds_default_windows():
    features = pd.DataFrame(_return_rows("AAPL", [pd.NA] + [0.01] * 70))

    result = add_volatility_features(features)

    for window in VOLATILITY_WINDOWS:
        assert f"realized_vol_{window}d" in result.columns


def test_add_volatility_features_requires_return_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        add_volatility_features(
            pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-01"]})
        )


def test_build_price_volatility_features_saves_output(tmp_path):
    feature_path = tmp_path / "price_return_features.parquet"
    output_path = tmp_path / "price_volatility_features.parquet"
    features = pd.DataFrame(_return_rows("AAPL", [pd.NA, 0.01, 0.02, 0.03]))
    features.to_parquet(feature_path, index=False)

    result = build_price_volatility_features(
        feature_path=feature_path,
        output_path=output_path,
        windows=[3],
        annualization_factor=1,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert saved.loc[3, "realized_vol_3d"] == pytest.approx(0.0081649658)
