import pandas as pd
import pytest

from market_qml.features.filing_events import (
    build_filing_event_feature_table,
    build_filing_event_features,
    merge_filing_event_features,
)


def _submissions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["aapl", "AAPL", "AAPL", "AAPL", "MSFT"],
            "cik": [320193, 320193, 320193, 320193, 789019],
            "cik_padded": [
                "0000320193",
                "0000320193",
                "0000320193",
                "0000320193",
                "0000789019",
            ],
            "form": ["10-K", "8-K", "10-Q", "8-K", "10-K"],
            "filing_date": [
                "2024-01-15",
                "2024-02-01",
                "2024-04-15",
                "2024-05-01",
                "2024-03-20",
            ],
            "report_date": [
                "2023-12-31",
                "",
                "2024-03-31",
                "",
                "2023-12-31",
            ],
            "accession_number": ["a-10k", "a-8k-1", "a-10q", "a-8k-2", "m-10k"],
            "primary_document": ["10k.htm", "8k1.htm", "10q.htm", "8k2.htm", "m10k.htm"],
        }
    )


def _market_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "AAPL", "MSFT", "SPY"],
            "date": pd.to_datetime(
                [
                    "2024-01-10",
                    "2024-01-20",
                    "2024-04-20",
                    "2024-05-10",
                    "2024-03-25",
                    "2024-05-10",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 103.0, 300.0, 400.0],
        }
    )


def test_build_filing_event_features_normalizes_and_filters_submissions():
    submissions = pd.concat(
        [
            _submissions(),
            pd.DataFrame(
                {
                    "symbol": ["AAPL"],
                    "cik": [320193],
                    "cik_padded": ["0000320193"],
                    "form": ["S-1"],
                    "filing_date": ["2024-06-01"],
                    "report_date": ["2024-05-31"],
                    "accession_number": ["a-s1"],
                    "primary_document": ["s1.htm"],
                }
            ),
        ],
        ignore_index=True,
    )

    result = build_filing_event_features(submissions)

    assert set(result["form"]) == {"10-K", "10-Q", "8-K"}
    assert result["symbol"].unique().tolist() == ["AAPL", "MSFT"]
    assert result.loc[0, "filing_date"] == pd.Timestamp("2024-01-15")


def test_merge_filing_event_features_uses_only_known_filings():
    events = build_filing_event_features(_submissions())

    result = merge_filing_event_features(_market_features(), events)
    aapl = result[result["symbol"] == "AAPL"].reset_index(drop=True)

    assert pd.isna(aapl.loc[0, "sec_last_filing_date"])
    assert pd.isna(aapl.loc[0, "sec_days_since_last_10k"])
    assert aapl.loc[1, "sec_last_filing_form"] == "10-K"
    assert aapl.loc[1, "sec_days_since_last_filing"] == 5
    assert aapl.loc[1, "sec_days_since_last_10k"] == 5
    assert pd.isna(aapl.loc[1, "sec_days_since_last_10q"])
    assert aapl.loc[2, "sec_last_filing_form"] == "10-Q"
    assert aapl.loc[2, "sec_days_since_last_10q"] == 5
    assert aapl.loc[2, "sec_days_since_last_8k"] == 79
    assert aapl.loc[3, "sec_last_filing_form"] == "8-K"
    assert aapl.loc[3, "sec_days_since_last_8k"] == 9


def test_merge_filing_event_features_adds_recent_and_form_indicators():
    events = build_filing_event_features(_submissions())

    result = merge_filing_event_features(_market_features(), events)
    row = result[(result["symbol"] == "AAPL") & (result["date"] == pd.Timestamp("2024-05-10"))].iloc[0]

    assert row["sec_recent_filing_30d"]
    assert row["sec_recent_8k_30d"]
    assert row["sec_recent_10q_90d"]
    assert not row["sec_recent_10k_90d"]
    assert row["sec_last_filing_is_8k"]
    assert not row["sec_last_filing_is_10k"]


def test_merge_filing_event_features_preserves_symbols_without_filings():
    events = build_filing_event_features(_submissions())

    result = merge_filing_event_features(_market_features(), events)
    spy = result[result["symbol"] == "SPY"].iloc[0]

    assert pd.isna(spy["sec_last_filing_date"])
    assert pd.isna(spy["sec_days_since_last_filing"])
    assert not spy["sec_recent_filing_30d"]
    assert not spy["sec_last_filing_is_10k"]


def test_build_filing_event_features_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        build_filing_event_features(pd.DataFrame({"symbol": ["AAPL"]}))


def test_build_filing_event_feature_table_saves_output(tmp_path):
    feature_path = tmp_path / "fundamental_features.parquet"
    submissions_path = tmp_path / "sec_submissions.parquet"
    output_path = tmp_path / "filing_event_features.parquet"

    _market_features().to_parquet(feature_path, index=False)
    _submissions().to_parquet(submissions_path, index=False)

    result = build_filing_event_feature_table(
        feature_path=feature_path,
        submissions_path=submissions_path,
        output_path=output_path,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert "sec_days_since_last_10k" in saved.columns
