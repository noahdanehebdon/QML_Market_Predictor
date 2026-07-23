import numpy as np
import pandas as pd
import pytest

from market_qml.features.regimes import build_market_regimes, save_market_regimes


def _features(rows=10):
    dates = pd.date_range("2024-01-01", periods=rows)
    spy_returns = [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.01, -0.01, 0.04, -0.04][
        :rows
    ]
    frames = []
    for symbol in ["AAPL", "SPY"]:
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "return_1d": spy_returns if symbol == "SPY" else [0.0] * rows,
                    "treasury_10y": np.linspace(4.0, 4.9, rows),
                    "treasury_2y": np.linspace(4.2, 4.3, rows),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_build_market_regimes_defines_volatility_rates_and_curve():
    result = build_market_regimes(
        _features(),
        volatility_window=3,
        rate_window=2,
        annualization_factor=1,
        minimum_threshold_history=2,
    )
    assert result.date.is_unique
    assert result.loc[4, "volatility_regime"] in {
        "high_volatility",
        "low_volatility",
        "normal_volatility",
    }
    assert result.loc[2, "rate_regime"] == "rising_rates"
    assert result.loc[0, "yield_curve_regime"] == "inverted_curve"
    assert result.loc[9, "yield_curve_regime"] == "normal_curve"
    assert result.loc[2, "yield_curve_trend"] == "steepening_curve"


def test_regimes_do_not_change_when_future_data_changes():
    original = _features()
    changed = original.copy()
    future = changed.date >= pd.Timestamp("2024-01-08")
    changed.loc[future & changed.symbol.eq("SPY"), "return_1d"] = 9.0
    changed.loc[future, ["treasury_10y", "treasury_2y"]] = [9.0, 1.0]
    kwargs = {
        "volatility_window": 3,
        "rate_window": 2,
        "annualization_factor": 1,
        "minimum_threshold_history": 2,
    }
    before = build_market_regimes(original, **kwargs).query("date < '2024-01-08'")
    after = build_market_regimes(changed, **kwargs).query("date < '2024-01-08'")
    pd.testing.assert_frame_equal(before, after)


def test_volatility_threshold_excludes_current_observation():
    features = _features()
    baseline = build_market_regimes(
        features,
        volatility_window=3,
        rate_window=2,
        annualization_factor=1,
        minimum_threshold_history=2,
    )
    changed = features.copy()
    changed.loc[
        (changed.symbol == "SPY") & (changed.date == "2024-01-07"), "return_1d"
    ] = 5.0
    stressed = build_market_regimes(
        changed,
        volatility_window=3,
        rate_window=2,
        annualization_factor=1,
        minimum_threshold_history=2,
    )
    assert (
        stressed.loc[6, "spy_realized_volatility"]
        != baseline.loc[6, "spy_realized_volatility"]
    )
    assert (
        stressed.loc[6, "spy_volatility_threshold"]
        == baseline.loc[6, "spy_volatility_threshold"]
    )


def test_market_regimes_validate_and_save(tmp_path):
    regimes = build_market_regimes(
        _features(),
        volatility_window=3,
        rate_window=2,
        annualization_factor=1,
        minimum_threshold_history=2,
    )
    output = tmp_path / "market_regimes.parquet"
    save_market_regimes(regimes, output)
    pd.testing.assert_frame_equal(pd.read_parquet(output), regimes)
    with pytest.raises(ValueError, match="missing columns"):
        build_market_regimes(pd.DataFrame({"date": ["2024-01-01"]}))
