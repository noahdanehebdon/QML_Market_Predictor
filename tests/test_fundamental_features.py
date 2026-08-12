import pandas as pd
import pytest

from market_qml.features.fundamentals import (
    build_filing_fundamental_features,
    build_fundamental_feature_table,
    merge_fundamental_features,
)


def _fundamental_rows() -> pd.DataFrame:
    rows = []
    filings = [
        ("AAPL", "0001", "2024-02-01", "2023", "FY", 100, 20, 200, 120, 80),
        ("AAPL", "0002", "2024-05-01", "2024", "Q1", 110, 22, 220, 130, 90),
        ("MSFT", "0003", "2024-03-01", "2023", "FY", 200, 50, 400, 160, 240),
    ]
    concepts = [
        ("revenue", 5),
        ("net_income", 6),
        ("assets", 7),
        ("liabilities", 8),
        ("stockholders_equity", 9),
    ]

    for symbol, accn, filing_date, fiscal_year, fiscal_period, *values in filings:
        for concept, value_index in concepts:
            rows.append(
                {
                    "symbol": symbol,
                    "ticker": symbol,
                    "cik": 320193 if symbol == "AAPL" else 789019,
                    "cik_padded": "0000320193" if symbol == "AAPL" else "0000789019",
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "filing_date": filing_date,
                    "form": "10-K" if fiscal_period == "FY" else "10-Q",
                    "concept": concept,
                    "taxonomy": "us-gaap",
                    "sec_concept": concept,
                    "value": values[value_index - 5],
                    "unit": "USD",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31" if fiscal_period == "FY" else "2024-03-31",
                    "accession_number": accn,
                    "frame": None,
                }
            )

    return pd.DataFrame(rows)


def _market_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT"],
            "date": pd.to_datetime(
                ["2024-01-15", "2024-02-15", "2024-05-15", "2024-03-15"]
            ),
            "close": [100, 101, 102, 200],
            "return_1d": [0.01, 0.02, 0.03, 0.04],
        }
    )


def test_build_filing_fundamental_features_pivots_and_derives_ratios():
    result = build_filing_fundamental_features(_fundamental_rows())
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)

    assert len(aapl) == 2
    assert aapl.loc[0, "fundamental_revenue"] == 100
    assert aapl.loc[0, "net_income_margin"] == pytest.approx(0.20)
    assert aapl.loc[0, "liability_ratio"] == pytest.approx(0.60)
    assert aapl.loc[0, "equity_ratio"] == pytest.approx(0.40)
    assert pd.isna(aapl.loc[0, "revenue_growth"])
    assert aapl.loc[1, "revenue_growth"] == pytest.approx(0.10)


def test_merge_fundamental_features_uses_filing_date_asof_without_leakage():
    filing_features = build_filing_fundamental_features(_fundamental_rows())

    result = merge_fundamental_features(_market_features(), filing_features)
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)

    assert pd.isna(aapl.loc[0, "fundamental_revenue"])
    assert aapl.loc[1, "fundamental_revenue"] == 100
    assert aapl.loc[1, "filing_recency_days"] == 14
    assert aapl.loc[2, "fundamental_revenue"] == 110
    assert aapl.loc[2, "filing_recency_days"] == 14


def test_merge_fundamentals_normalizes_mixed_timestamp_resolutions():
    market = _market_features().astype({"date": "datetime64[s]"})
    filings = build_filing_fundamental_features(_fundamental_rows())
    filings["filing_date"] = filings["filing_date"].astype("datetime64[us]")

    result = merge_fundamental_features(market, filings)

    assert result["date"].dtype == "datetime64[ns]"
    assert result["filing_date"].dtype == "datetime64[ns]"


def test_merge_fundamental_features_preserves_symbols_without_filings():
    filing_features = build_filing_fundamental_features(_fundamental_rows())
    market = pd.DataFrame(
        {
            "symbol": ["NVDA"],
            "date": [pd.Timestamp("2024-06-01")],
            "close": [900],
        }
    )

    result = merge_fundamental_features(market, filing_features)

    assert len(result) == 1
    assert pd.isna(result.loc[0, "fundamental_revenue"])
    assert pd.isna(result.loc[0, "filing_recency_days"])


def test_build_filing_fundamental_features_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        build_filing_fundamental_features(pd.DataFrame({"symbol": ["AAPL"]}))


def test_build_fundamental_feature_table_saves_output(tmp_path):
    feature_path = tmp_path / "macro_features.parquet"
    fundamentals_path = tmp_path / "fundamentals.parquet"
    output_path = tmp_path / "fundamental_features.parquet"

    _market_features().to_parquet(feature_path, index=False)
    _fundamental_rows().to_parquet(fundamentals_path, index=False)

    result = build_fundamental_feature_table(
        feature_path=feature_path,
        fundamentals_path=fundamentals_path,
        output_path=output_path,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert "net_income_margin" in saved.columns
