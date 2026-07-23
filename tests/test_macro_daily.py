import pandas as pd

from scripts.build_macro_daily import build_macro_daily


def test_build_macro_daily_lags_monthly_macro_before_forward_fill(tmp_path):
    price_path = tmp_path / "prices.parquet"
    macro_path = tmp_path / "macro.parquet"
    output_path = tmp_path / "macro_daily.parquet"

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"]),
            "symbol": ["SPY", "SPY", "SPY"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    prices.to_parquet(price_path, index=False)

    macro = pd.DataFrame(
        {
            "treasury_10y": [4.0],
            "treasury_2y": [4.2],
            "fed_funds": [5.3],
            "cpi_all_items_sa": [310.0],
            "unemployment_rate": [3.8],
            "industrial_production": [102.5],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )
    macro.to_parquet(macro_path)

    result = build_macro_daily(
        price_path=price_path,
        macro_path=macro_path,
        output_path=output_path,
        lag_daily_rates=False,
    )

    assert output_path.exists()
    assert list(result.index) == list(pd.to_datetime(prices["date"]))
    assert pd.isna(result.loc[pd.Timestamp("2024-01-31"), "cpi_all_items_sa"])
    assert result.loc[pd.Timestamp("2024-02-01"), "cpi_all_items_sa"] == 310.0
    assert result.loc[pd.Timestamp("2024-02-02"), "unemployment_rate"] == 3.8
