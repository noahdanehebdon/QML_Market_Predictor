import pandas as pd
import pytest

from market_qml.models.dataset import (
    build_modeling_dataset,
    build_train_validation_datasets,
    load_modeling_dataset,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "close": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
            "return_1d": [None, 0.01, 0.0099, None, 0.005, 0.005],
            "realized_vol_5d": [0.1, 0.11, None, 0.2, 0.21, 0.22],
            "sec_recent_filing_30d": [False, True, True, False, False, True],
            "filing_date": pd.to_datetime(
                [
                    pd.NaT,
                    "2023-12-15",
                    "2023-12-15",
                    pd.NaT,
                    pd.NaT,
                    "2023-12-20",
                ]
            ),
            "form": [pd.NA, "10-K", "10-K", pd.NA, pd.NA, "8-K"],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["aapl", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "label_horizon_days": [5, 5, 5, 5, 5, 5],
            "forward_return_5d": [0.05, 0.04, pd.NA, -0.01, 0.02, 0.03],
            "spy_forward_return_5d": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "forward_excess_return_5d": [0.04, 0.03, pd.NA, -0.02, 0.01, 0.02],
            "outperform_spy_5d": [1, 1, pd.NA, 0, 1, 1],
        }
    )


def test_build_modeling_dataset_returns_X_y_and_metadata_without_label_leakage():
    dataset = build_modeling_dataset(_features(), _labels())

    assert list(dataset.X.columns) == [
        "close",
        "return_1d",
        "realized_vol_5d",
        "sec_recent_filing_30d",
    ]
    assert dataset.y.tolist() == [1, 1, 0, 1, 1]
    assert dataset.metadata[["symbol", "date"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2024-01-01")],
        ["AAPL", pd.Timestamp("2024-01-02")],
        ["MSFT", pd.Timestamp("2024-01-01")],
        ["MSFT", pd.Timestamp("2024-01-02")],
        ["MSFT", pd.Timestamp("2024-01-03")],
    ]
    assert "forward_return_5d" not in dataset.X.columns
    assert "forward_excess_return_5d" not in dataset.X.columns
    assert "forward_excess_return_5d" in dataset.metadata.columns


def test_build_modeling_dataset_supports_date_filters():
    dataset = build_modeling_dataset(
        _features(),
        _labels(),
        start_date="2024-01-02",
        end_date="2024-01-02",
    )

    assert dataset.metadata["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
    assert dataset.metadata["symbol"].tolist() == ["AAPL", "MSFT"]


def test_build_modeling_dataset_supports_requested_feature_columns():
    dataset = build_modeling_dataset(
        _features(),
        _labels(),
        feature_columns=["close", "sec_recent_filing_30d"],
    )

    assert list(dataset.X.columns) == ["close", "sec_recent_filing_30d"]


def test_build_modeling_dataset_drops_rows_with_too_many_missing_features():
    dataset = build_modeling_dataset(
        _features(),
        _labels(),
        feature_columns=["close", "return_1d", "realized_vol_5d"],
        max_missing_feature_fraction=0,
    )

    assert dataset.metadata[["symbol", "date"]].values.tolist() == [
        ["AAPL", pd.Timestamp("2024-01-02")],
        ["MSFT", pd.Timestamp("2024-01-02")],
        ["MSFT", pd.Timestamp("2024-01-03")],
    ]


def test_build_train_validation_datasets_uses_matching_feature_columns():
    result = build_train_validation_datasets(
        _features(),
        _labels(),
        train_start_date="2024-01-01",
        train_end_date="2024-01-01",
        validation_start_date="2024-01-02",
        validation_end_date="2024-01-03",
    )

    assert result.train.metadata["date"].unique().tolist() == [pd.Timestamp("2024-01-01")]
    assert result.validation.metadata["date"].min() == pd.Timestamp("2024-01-02")
    assert list(result.train.X.columns) == list(result.validation.X.columns)


def test_build_modeling_dataset_validates_inputs():
    with pytest.raises(ValueError, match="Label table is missing required columns"):
        build_modeling_dataset(_features(), pd.DataFrame({"symbol": ["AAPL"]}))

    duplicate_labels = pd.concat([_labels(), _labels().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        build_modeling_dataset(_features(), duplicate_labels)

    with pytest.raises(ValueError, match="max_missing_feature_fraction"):
        build_modeling_dataset(_features(), _labels(), max_missing_feature_fraction=2)


def test_load_modeling_dataset_reads_parquet_inputs(tmp_path):
    feature_path = tmp_path / "feature_table.parquet"
    label_path = tmp_path / "forward_return_labels.parquet"
    _features().to_parquet(feature_path, index=False)
    _labels().to_parquet(label_path, index=False)

    dataset = load_modeling_dataset(
        feature_path=feature_path,
        label_path=label_path,
        feature_columns=["close"],
    )

    assert list(dataset.X.columns) == ["close"]
    assert len(dataset.y) == 5
