import pandas as pd
import pytest

from market_qml.backtest.splits import (
    build_walk_forward_split_table,
    generate_walk_forward_splits,
)


def _metadata(dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    if dates is None:
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for symbol in ["AAPL", "MSFT"]:
        for date in dates:
            rows.append({"symbol": symbol, "date": date})
    return pd.DataFrame(rows)


def test_generate_walk_forward_splits_uses_train_dates_before_validation_dates():
    result = generate_walk_forward_splits(
        _metadata(),
        train_window_days=4,
        validation_window_days=2,
        step_days=2,
    )

    assert len(result) == 3
    first = result.iloc[0]
    assert first["train_start_date"] == pd.Timestamp("2024-01-01")
    assert first["train_end_date"] == pd.Timestamp("2024-01-04")
    assert first["validation_start_date"] == pd.Timestamp("2024-01-05")
    assert first["validation_end_date"] == pd.Timestamp("2024-01-06")
    assert first["train_days"] == 4
    assert first["validation_days"] == 2
    assert first["train_rows"] == 8
    assert first["validation_rows"] == 4
    assert (result["train_end_date"] < result["validation_start_date"]).all()


def test_generate_walk_forward_splits_validation_periods_do_not_overlap_training():
    result = generate_walk_forward_splits(
        _metadata(),
        train_window_days=4,
        validation_window_days=2,
        step_days=1,
    )

    for row in result.itertuples(index=False):
        train_dates = pd.date_range(row.train_start_date, row.train_end_date, freq="D")
        validation_dates = pd.date_range(
            row.validation_start_date,
            row.validation_end_date,
            freq="D",
        )
        assert set(train_dates).isdisjoint(set(validation_dates))


def test_generate_walk_forward_splits_supports_yearly_validation_windows():
    dates = pd.bdate_range("2020-12-15", "2022-12-31")

    result = generate_walk_forward_splits(
        _metadata(dates),
        train_window_days=5,
        validation_window_days=252,
        yearly_validation=True,
    )

    assert result["validation_start_date"].dt.year.tolist() == [2021, 2022]
    assert result["validation_end_date"].dt.year.tolist() == [2021, 2022]
    assert result["train_days"].tolist() == [5, 5]
    assert (result["train_end_date"] < result["validation_start_date"]).all()


def test_generate_walk_forward_splits_returns_empty_when_history_is_too_short():
    result = generate_walk_forward_splits(
        _metadata(pd.date_range("2024-01-01", periods=3, freq="D")),
        train_window_days=4,
        validation_window_days=2,
    )

    assert result.empty
    assert list(result.columns) == [
        "split_id",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "train_days",
        "validation_days",
        "train_rows",
        "validation_rows",
    ]


def test_generate_walk_forward_splits_validates_inputs():
    with pytest.raises(ValueError, match="train_window_days"):
        generate_walk_forward_splits(_metadata(), train_window_days=0)

    with pytest.raises(ValueError, match="date column"):
        generate_walk_forward_splits(pd.DataFrame({"not_date": ["2024-01-01"]}))


def test_build_walk_forward_split_table_saves_output(tmp_path):
    output_path = tmp_path / "walk_forward_splits.parquet"

    result = build_walk_forward_split_table(
        _metadata(),
        output_path=output_path,
        train_window_days=4,
        validation_window_days=2,
    )
    saved = pd.read_parquet(output_path)

    assert output_path.exists()
    assert list(saved.columns) == list(result.columns)
    assert len(saved) == len(result)
