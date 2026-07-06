import pandas as pd
import pytest

from market_qml.features.canonical import (
    build_canonical_feature_table,
    build_canonical_features,
)


def _feature_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["msft", "AAPL", "AAPL", "MSFT"],
            "date": pd.to_datetime(
                ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02"]
            ),
            "close": [210.0, 100.0, 101.0, 209.0],
            "return_1d": [0.01, 0.02, 0.01, -0.01],
            "excess_return_1d_vs_spy": [0.004, 0.01, 0.005, -0.02],
            "treasury_10y": [4.0, 4.0, 4.1, 4.0],
            "fundamental_assets": [500.0, 300.0, 300.0, 500.0],
            "sec_days_since_last_10k": [20.0, 10.0, 11.0, 19.0],
        }
    )


def test_build_canonical_features_sorts_and_keys_symbol_date_rows():
    result = build_canonical_features(_feature_rows())

    assert result[["symbol", "date"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2024-01-02")],
        ["AAPL", pd.Timestamp("2024-01-03")],
        ["MSFT", pd.Timestamp("2024-01-02")],
        ["MSFT", pd.Timestamp("2024-01-03")],
    ]
    assert not result.duplicated(subset=["symbol", "date"]).any()
    assert "excess_return_1d_vs_spy" in result.columns


def test_build_canonical_features_rejects_label_columns():
    features = _feature_rows()
    features["forward_return_5d"] = 0.02

    with pytest.raises(ValueError, match="contains label columns"):
        build_canonical_features(features)


def test_build_canonical_features_rejects_duplicate_keys():
    features = pd.concat([_feature_rows(), _feature_rows().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate symbol/date"):
        build_canonical_features(features)


def test_build_canonical_features_requires_key_schema():
    with pytest.raises(ValueError, match="missing required key columns"):
        build_canonical_features(pd.DataFrame({"symbol": ["AAPL"], "close": [100.0]}))


def test_build_canonical_feature_table_saves_output(tmp_path):
    feature_path = tmp_path / "filing_event_features.parquet"
    output_path = tmp_path / "feature_table.parquet"
    _feature_rows().to_parquet(feature_path, index=False)

    result = build_canonical_feature_table(
        feature_path=feature_path,
        output_path=output_path,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert saved[["symbol", "date"]].equals(result[["symbol", "date"]])
