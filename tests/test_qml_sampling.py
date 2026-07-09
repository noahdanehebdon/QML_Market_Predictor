import pandas as pd
import pytest

from market_qml.qml.sampling import (
    build_qml_sample,
    load_qml_features,
    save_qml_sample,
    save_qml_sample_metadata,
)


def _qml_features() -> pd.DataFrame:
    rows = []
    for role, dates in [
        ("train", pd.date_range("2024-01-01", periods=6, freq="D")),
        ("validation", pd.date_range("2024-02-01", periods=4, freq="D")),
    ]:
        for date_index, date in enumerate(dates):
            for symbol_index, symbol in enumerate(["AAPL", "MSFT", "NVDA", "AMZN"]):
                target = int((date_index + symbol_index) % 2 == 0)
                rows.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "split_id": 0,
                        "sample_role": role,
                        "target": target,
                        "pca_00": float(date_index),
                        "pca_01": float(symbol_index),
                    }
                )
    return pd.DataFrame(rows)


def test_build_qml_sample_limits_rows_balances_classes_and_preserves_time_order():
    result = build_qml_sample(
        _qml_features(),
        max_train_rows_per_split=8,
        max_validation_rows_per_split=4,
        balance_classes=True,
        random_seed=7,
    )

    sample = result.sample
    train = sample[sample["sample_role"] == "train"]
    validation = sample[sample["sample_role"] == "validation"]

    assert len(train) == 8
    assert len(validation) == 4
    assert train["target"].value_counts().sort_index().tolist() == [4, 4]
    assert validation["target"].value_counts().sort_index().tolist() == [2, 2]
    assert sample.equals(sample.sort_values(["split_id", "sample_role", "date", "symbol"]))
    assert result.metadata["sampled_rows"].tolist() == [8, 4]
    assert result.metadata["balanced_classes"].tolist() == [True, True]
    assert result.metadata["rows_after_date_filter"].tolist() == [24, 16]
    assert result.metadata["rows_after_balance"].tolist() == [24, 16]


def test_build_qml_sample_is_reproducible_for_same_seed():
    first = build_qml_sample(
        _qml_features(),
        max_train_rows_per_split=6,
        max_validation_rows_per_split=4,
        max_dates_per_split_role=3,
        max_symbols=3,
        random_seed=123,
    )
    second = build_qml_sample(
        _qml_features(),
        max_train_rows_per_split=6,
        max_validation_rows_per_split=4,
        max_dates_per_split_role=3,
        max_symbols=3,
        random_seed=123,
    )

    pd.testing.assert_frame_equal(first.sample, second.sample)
    pd.testing.assert_frame_equal(first.metadata, second.metadata)


def test_build_qml_sample_skips_balancing_for_continuous_targets():
    features = _qml_features()
    features["target"] = range(len(features))

    result = build_qml_sample(
        features,
        max_train_rows_per_split=5,
        max_validation_rows_per_split=3,
        balance_classes=True,
        random_seed=7,
    )

    assert result.metadata["balanced_classes"].tolist() == [False, False]
    assert result.metadata["rows_after_balance"].tolist() == [24, 16]
    assert result.metadata["sampled_rows"].tolist() == [5, 3]


def test_build_qml_sample_validates_inputs():
    with pytest.raises(ValueError, match="missing required columns"):
        build_qml_sample(pd.DataFrame({"symbol": ["AAPL"]}))

    with pytest.raises(ValueError, match="PCA component"):
        build_qml_sample(
            _qml_features().drop(columns=["pca_00", "pca_01"]),
        )

    with pytest.raises(ValueError, match="positive"):
        build_qml_sample(_qml_features(), max_train_rows_per_split=0)


def test_qml_sample_outputs_can_be_saved(tmp_path):
    result = build_qml_sample(
        _qml_features(),
        max_train_rows_per_split=8,
        max_validation_rows_per_split=4,
        random_seed=7,
    )
    sample_path = tmp_path / "qml_sample.parquet"
    metadata_path = tmp_path / "qml_sample_metadata.parquet"

    save_qml_sample(result.sample, sample_path)
    save_qml_sample_metadata(result.metadata, metadata_path)
    loaded = load_qml_features(sample_path)
    saved_metadata = pd.read_parquet(metadata_path)

    assert sample_path.exists()
    assert metadata_path.exists()
    assert list(loaded.columns) == list(result.sample.columns)
    assert saved_metadata["sampled_rows"].tolist() == result.metadata[
        "sampled_rows"
    ].tolist()
