import pandas as pd
import pytest

from market_qml.features.market_signals import add_market_signal_features


def _prices(periods: int = 70) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods)
    rows = []
    for symbol, drift in [("SPY", 0.001), ("AAA", 0.002)]:
        close = 100.0
        for index, date in enumerate(dates):
            daily_return = drift + ((index % 5) - 2) * 0.0005
            close *= 1 + daily_return
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": 1_000_000 + index * 100,
                    "return_1d": daily_return,
                    "return_5d": daily_return * 5,
                }
            )
    return pd.DataFrame(rows)


def test_market_signals_add_stationary_trailing_features():
    result = add_market_signal_features(_prices())
    aaa = result.loc[result["symbol"].eq("AAA")].iloc[-1]

    assert aaa["residual_momentum_20d"] > 0
    assert -1 <= aaa["drawdown_60d"] <= 0
    assert aaa["amihud_illiquidity_20d"] > 0
    assert 0 <= aaa["positive_day_share_20d"] <= 1
    assert aaa["reversal_5d"] < 0


def test_market_signals_do_not_change_when_future_rows_are_mutated():
    prices = _prices()
    baseline = add_market_signal_features(prices)
    cutoff = pd.Timestamp("2024-02-20")
    mutated = prices.copy()
    future = mutated["date"] > cutoff
    mutated.loc[future, ["close", "high", "low", "volume", "return_1d"]] = [
        99999.0,
        100000.0,
        99998.0,
        1.0,
        9.0,
    ]
    changed = add_market_signal_features(mutated)
    columns = [column for column in baseline if column not in prices]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["date"].le(cutoff), columns].reset_index(drop=True),
        changed.loc[changed["date"].le(cutoff), columns].reset_index(drop=True),
    )


def test_market_signals_require_a_benchmark():
    with pytest.raises(ValueError, match="SPY"):
        add_market_signal_features(_prices().query("symbol == 'AAA'"))
