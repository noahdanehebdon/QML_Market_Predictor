import pandas as pd
import pytest

from market_qml.features.volume import (
    VOLUME_WINDOWS,
    add_volume_features,
    build_price_volume_features,
)


def _feature_rows(symbol: str, volumes: list[float]) -> list[dict]:
    return [
        {
            "symbol": symbol,
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i),
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "open": 99 + i,
            "high": 101 + i,
            "low": 98 + i,
            "close": 100 + i,
            "volume": volume,
            "trade_count": 100 + i,
            "vwap": 100 + i,
            "return_1d": pd.NA,
            "realized_vol_5d": pd.NA,
        }
        for i, volume in enumerate(volumes)
    ]


def test_add_volume_features_preserves_existing_columns_and_adds_volume_features():
    features = pd.DataFrame(_feature_rows("AAPL", [100, 110, 120, 130]))

    result = add_volume_features(features, windows=[3])

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
        "realized_vol_5d",
        "dollar_volume",
        "avg_volume_3d",
        "volume_shock_3d",
        "avg_dollar_volume_3d",
    ]
    assert result.loc[0, "dollar_volume"] == 100 * 100
    assert result["avg_volume_3d"].iloc[:2].isna().all()
    assert result.loc[2, "avg_volume_3d"] == pytest.approx(110)
    assert pd.isna(result.loc[2, "volume_shock_3d"])
    assert result.loc[3, "volume_shock_3d"] == pytest.approx((130 / 110) - 1)


def test_add_volume_features_computes_by_symbol_without_leakage():
    features = pd.DataFrame(
        _feature_rows("AAPL", [100, 110, 120, 130])
        + _feature_rows("MSFT", [1000, 900, 800, 700])
    )
    features = features.sort_values(["date", "symbol"], ascending=[True, False])

    result = add_volume_features(features, windows=[3])
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    msft = result[result["symbol"] == "MSFT"].reset_index(drop=True)

    assert aapl.loc[2, "avg_volume_3d"] == pytest.approx(110)
    assert msft.loc[2, "avg_volume_3d"] == pytest.approx(900)
    assert aapl.loc[3, "volume_shock_3d"] == pytest.approx((130 / 110) - 1)
    assert msft.loc[3, "volume_shock_3d"] == pytest.approx((700 / 900) - 1)


def test_add_volume_features_adds_default_windows():
    features = pd.DataFrame(_feature_rows("AAPL", list(range(100, 170))))

    result = add_volume_features(features)

    for window in VOLUME_WINDOWS:
        assert f"avg_volume_{window}d" in result.columns
        assert f"volume_shock_{window}d" in result.columns
        assert f"avg_dollar_volume_{window}d" in result.columns


def test_add_volume_features_adds_optional_liquidity_filter():
    features = pd.DataFrame(_feature_rows("AAPL", [100, 110, 120, 130]))

    result = add_volume_features(
        features,
        windows=[3],
        liquidity_min_avg_dollar_volume=10_000,
        liquidity_window=3,
    )

    assert "is_liquid_3d" in result.columns
    assert bool(result.loc[2, "is_liquid_3d"])


def test_add_volume_features_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        add_volume_features(pd.DataFrame({"symbol": ["AAPL"], "date": ["2024-01-01"]}))


def test_build_price_volume_features_saves_output(tmp_path):
    feature_path = tmp_path / "price_volatility_features.parquet"
    output_path = tmp_path / "price_volume_features.parquet"
    features = pd.DataFrame(_feature_rows("AAPL", [100, 110, 120, 130]))
    features.to_parquet(feature_path, index=False)

    result = build_price_volume_features(
        feature_path=feature_path,
        output_path=output_path,
        windows=[3],
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert saved.loc[3, "volume_shock_3d"] == pytest.approx((130 / 110) - 1)
