import pandas as pd
import pytest

from scripts.generate_demo_prices import DEMO_SYMBOLS, generate_demo_prices


def test_demo_prices_are_deterministic_and_market_like():
    first = generate_demo_prices(days=400, seed=7)
    second = generate_demo_prices(days=400, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["symbol"]) == set(DEMO_SYMBOLS)
    assert first.groupby("symbol")["date"].nunique().eq(400).all()
    assert first["close"].gt(0).all()
    assert first[["symbol", "date"]].duplicated().sum() == 0


def test_demo_prices_require_enough_history_for_locked_validation():
    with pytest.raises(ValueError, match="at least 400"):
        generate_demo_prices(days=399)
