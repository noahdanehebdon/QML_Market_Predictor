import pandas as pd
import pytest

from market_qml.features.benchmark import (
    add_benchmark_relative_features,
    build_benchmark_relative_features,
)


def _feature_rows(symbol: str, returns: list[float], offset: float = 0.0) -> list[dict]:
    closes = [100.0 + offset]
    for daily_return in returns[1:]:
        closes.append(closes[-1] * (1 + daily_return))

    return [
        {
            "symbol": symbol,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "close": closes[i],
            "volume": 1000 + i,
            "return_1d": returns[i],
            "return_3d": pd.NA,
            "return_5d": pd.NA,
            "return_20d": pd.NA,
            "return_60d": pd.NA,
            "realized_vol_3d": 0.10 + offset,
            "realized_vol_20d": 0.20 + offset,
            "realized_vol_60d": 0.30 + offset,
            "dollar_volume": closes[i] * (1000 + i),
        }
        for i in range(len(returns))
    ]


def test_add_benchmark_relative_features_adds_excess_returns():
    features = pd.DataFrame(
        _feature_rows("AAPL", [pd.NA, 0.03, 0.04])
        + _feature_rows("SPY", [pd.NA, 0.01, 0.02])
    )
    features.loc[features["symbol"] == "AAPL", "return_5d"] = [pd.NA, pd.NA, 0.20]
    features.loc[features["symbol"] == "SPY", "return_5d"] = [pd.NA, pd.NA, 0.10]

    result = add_benchmark_relative_features(
        features,
        benchmark_symbol="SPY",
        windows=[2],
        excess_return_windows=[1, 5],
    )
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    spy = result[result["symbol"] == "SPY"].reset_index(drop=True)

    assert aapl.loc[1, "excess_return_1d_vs_spy"] == pytest.approx(0.02)
    assert aapl.loc[2, "excess_return_5d_vs_spy"] == pytest.approx(0.10)
    assert spy.loc[1, "excess_return_1d_vs_spy"] == pytest.approx(0.0)


def test_add_benchmark_relative_features_computes_corr_beta_and_relative_metrics():
    features = pd.DataFrame(
        _feature_rows("AAPL", [pd.NA, 0.02, 0.04, 0.06])
        + _feature_rows("SPY", [pd.NA, 0.01, 0.02, 0.03])
    )
    features.loc[features["symbol"] == "AAPL", "return_20d"] = [
        pd.NA,
        pd.NA,
        pd.NA,
        0.12,
    ]
    features.loc[features["symbol"] == "SPY", "return_20d"] = [
        pd.NA,
        pd.NA,
        pd.NA,
        0.06,
    ]
    features.loc[features["symbol"] == "AAPL", "return_3d"] = [
        pd.NA,
        pd.NA,
        pd.NA,
        0.12,
    ]
    features.loc[features["symbol"] == "SPY", "return_3d"] = [pd.NA, pd.NA, pd.NA, 0.06]

    result = add_benchmark_relative_features(
        features,
        benchmark_symbol="SPY",
        windows=[3],
        excess_return_windows=[20],
    )
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)
    spy = result[result["symbol"] == "SPY"].reset_index(drop=True)

    assert aapl["rolling_corr_3d_vs_spy"].iloc[:3].isna().all()
    assert aapl.loc[3, "rolling_corr_3d_vs_spy"] == pytest.approx(1.0)
    assert aapl.loc[3, "rolling_beta_3d_vs_spy"] == pytest.approx(2.0)
    assert aapl.loc[3, "relative_vol_3d_vs_spy"] == pytest.approx(1.0)
    assert aapl.loc[3, "relative_momentum_3d_vs_spy"] == pytest.approx(0.06)
    assert spy.loc[3, "rolling_beta_3d_vs_spy"] == pytest.approx(1.0)


def test_add_benchmark_relative_features_requires_benchmark_rows():
    features = pd.DataFrame(_feature_rows("AAPL", [pd.NA, 0.01, 0.02]))

    with pytest.raises(ValueError, match="No benchmark rows found"):
        add_benchmark_relative_features(features, benchmark_symbol="SPY")


def test_add_benchmark_relative_features_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        add_benchmark_relative_features(pd.DataFrame({"symbol": ["AAPL"]}))


def test_build_benchmark_relative_features_saves_output(tmp_path):
    feature_path = tmp_path / "price_volume_features.parquet"
    output_path = tmp_path / "benchmark_relative_features.parquet"
    features = pd.DataFrame(
        _feature_rows("AAPL", [pd.NA, 0.03, 0.04])
        + _feature_rows("SPY", [pd.NA, 0.01, 0.02])
    )
    features.to_parquet(feature_path, index=False)

    result = build_benchmark_relative_features(
        feature_path=feature_path,
        output_path=output_path,
        benchmark_symbol="SPY",
        windows=[2],
        excess_return_windows=[1],
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert "excess_return_1d_vs_spy" in saved.columns
