import pandas as pd
import pytest

from market_qml.features.fundamentals import (
    build_filing_fundamental_features,
    merge_fundamental_features,
)
from market_qml.features.returns import add_return_features
from market_qml.features.volatility import add_volatility_features
from market_qml.models.dataset import build_modeling_dataset
from scripts.build_macro_daily import (
    build_daily_rate_features,
    build_monthly_macro_features,
)


def test_return_features_do_not_use_future_prices():
    prices = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "close": [100.0, 110.0, 121.0, 133.1, 9999.0],
        }
    )

    result = add_return_features(prices, windows=[1, 3])

    assert result.loc[1, "return_1d"] == pytest.approx(0.10)
    assert result.loc[2, "return_1d"] == pytest.approx(0.10)
    assert result.loc[3, "return_3d"] == pytest.approx(0.331)


def test_volatility_features_do_not_use_future_returns():
    features = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5,
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "return_1d": [0.01, 0.02, 0.03, 10.0, -10.0],
        }
    )

    result = add_volatility_features(features, windows=[3], annualization_factor=1)

    expected = pd.Series([0.01, 0.02, 0.03]).std(ddof=0)
    assert pd.isna(result.loc[1, "realized_vol_3d"])
    assert result.loc[2, "realized_vol_3d"] == pytest.approx(expected)


def test_macro_monthly_values_are_unavailable_before_safe_release_date():
    macro = pd.DataFrame(
        {
            "treasury_10y": [4.0],
            "treasury_2y": [3.8],
            "fed_funds": [5.25],
            "cpi_all_items_sa": [300.0],
            "unemployment_rate": [3.8],
            "industrial_production": [100.0],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )
    trading_dates = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-02-02"])

    result = build_monthly_macro_features(macro, trading_dates)

    assert pd.isna(result.loc[pd.Timestamp("2024-01-31"), "cpi_all_items_sa"])
    assert result.loc[pd.Timestamp("2024-02-01"), "cpi_all_items_sa"] == 300.0
    assert result.loc[pd.Timestamp("2024-02-02"), "unemployment_rate"] == 3.8


def test_daily_macro_lag_mode_blocks_same_day_rate_observations():
    macro = pd.DataFrame(
        {
            "treasury_10y": [4.0, 9.9],
            "treasury_2y": [3.8, 9.8],
            "fed_funds": [5.25, 9.7],
            "cpi_all_items_sa": [300.0, 301.0],
            "unemployment_rate": [3.8, 3.9],
            "industrial_production": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    trading_dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])

    result = build_daily_rate_features(
        macro=macro,
        trading_dates=trading_dates,
        lag_daily_rates=True,
    )

    assert pd.isna(result.loc[pd.Timestamp("2024-01-01"), "treasury_10y"])
    assert result.loc[pd.Timestamp("2024-01-02"), "treasury_10y"] == 4.0
    assert result.loc[pd.Timestamp("2024-01-03"), "treasury_10y"] == 9.9


def test_fundamentals_are_known_only_after_filing_date():
    fundamentals = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "cik": [320193, 320193],
            "cik_padded": ["0000320193", "0000320193"],
            "fiscal_year": [2023, 2024],
            "fiscal_period": ["FY", "FY"],
            "filing_date": pd.to_datetime(["2024-01-31", "2024-03-01"]),
            "form": ["10-K", "10-K"],
            "concept": ["revenue", "revenue"],
            "value": [100.0, 9999.0],
            "end_date": pd.to_datetime(["2023-12-31", "2024-12-31"]),
            "accession_number": ["old", "future"],
        }
    )
    market = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-30", "2024-02-15", "2024-03-01"]),
            "close": [10.0, 11.0, 12.0],
        }
    )

    filing_features = build_filing_fundamental_features(fundamentals)
    result = merge_fundamental_features(market, filing_features)

    assert pd.isna(result.loc[0, "fundamental_revenue"])
    assert result.loc[1, "fundamental_revenue"] == 100.0
    assert result.loc[1, "filing_date"] == pd.Timestamp("2024-01-31")
    assert result.loc[2, "fundamental_revenue"] == 9999.0


def test_modeling_dataset_excludes_all_label_columns_from_features():
    features = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "close": [100.0, 101.0],
            "return_1d": [0.01, 0.01],
        }
    )
    labels = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "label_horizon_days": [5, 5],
            "forward_return_5d": [999.0, 888.0],
            "spy_forward_return_5d": [0.01, 0.01],
            "forward_excess_return_5d": [998.99, 887.99],
            "outperform_spy_5d": [1, 1],
        }
    )

    dataset = build_modeling_dataset(features, labels)

    assert list(dataset.X.columns) == ["close", "return_1d"]
    assert "forward_return_5d" not in dataset.X.columns
    assert "forward_excess_return_5d" not in dataset.X.columns
    assert "label_horizon_days" not in dataset.X.columns
