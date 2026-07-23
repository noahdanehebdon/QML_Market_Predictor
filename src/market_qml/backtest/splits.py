"""Walk-forward validation split generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_qml.backtest.validation import PROTOCOL_VERSION, partition_locked_test

REQUIRED_DATE_COLUMN = "date"
DEFAULT_SPLIT_OUTPUT_PATH = Path("data/processed/walk_forward_splits.parquet")


def generate_walk_forward_splits(
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    date_column: str = REQUIRED_DATE_COLUMN,
    train_window_days: int = 756,
    validation_window_days: int = 126,
    step_days: int | None = None,
    yearly_validation: bool = False,
    purge_days: int = 0,
    locked_test_days: int = 0,
    embargo_days: int = 0,
) -> pd.DataFrame:
    """Generate time-ordered walk-forward split metadata.

    Splits are defined over sorted unique trading dates. Training windows always
    end before validation windows begin, and random shuffling is not supported.
    """
    _validate_positive_window(train_window_days, "train_window_days")
    _validate_positive_window(validation_window_days, "validation_window_days")
    if step_days is not None:
        _validate_positive_window(step_days, "step_days")
    if purge_days < 0:
        raise ValueError("purge_days cannot be negative.")

    locked_manifest = None
    if locked_test_days:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Locked-test partitioning requires a DataFrame.")
        data, _, locked_manifest = partition_locked_test(
            data,
            locked_test_days=locked_test_days,
            embargo_days=embargo_days,
            date_column=date_column,
        )
    dates = _unique_dates(data, date_column=date_column)
    if len(dates) <= train_window_days:
        return _empty_split_metadata()

    if yearly_validation:
        result = _generate_yearly_splits(
            data=data,
            dates=dates,
            train_window_days=train_window_days,
            date_column=date_column,
            purge_days=purge_days,
        )
        return _attach_locked_test_metadata(result, locked_manifest)

    result = _generate_fixed_window_splits(
        data=data,
        dates=dates,
        train_window_days=train_window_days,
        validation_window_days=validation_window_days,
        step_days=step_days or validation_window_days,
        date_column=date_column,
        purge_days=purge_days,
    )
    return _attach_locked_test_metadata(result, locked_manifest)


def save_walk_forward_splits(splits: pd.DataFrame, output_path: str | Path) -> None:
    """Save split metadata to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(output_path, index=False)


def build_walk_forward_split_table(
    metadata: pd.DataFrame,
    output_path: str | Path = DEFAULT_SPLIT_OUTPUT_PATH,
    *,
    date_column: str = REQUIRED_DATE_COLUMN,
    train_window_days: int = 756,
    validation_window_days: int = 126,
    step_days: int | None = None,
    yearly_validation: bool = False,
    purge_days: int = 0,
    locked_test_days: int = 0,
    embargo_days: int = 0,
) -> pd.DataFrame:
    """Generate and save walk-forward split metadata from row metadata."""
    splits = generate_walk_forward_splits(
        metadata,
        date_column=date_column,
        train_window_days=train_window_days,
        validation_window_days=validation_window_days,
        step_days=step_days,
        yearly_validation=yearly_validation,
        purge_days=purge_days,
        locked_test_days=locked_test_days,
        embargo_days=embargo_days,
    )
    save_walk_forward_splits(splits, output_path)
    return splits


def _generate_fixed_window_splits(
    *,
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    dates: pd.DatetimeIndex,
    train_window_days: int,
    validation_window_days: int,
    step_days: int,
    date_column: str,
    purge_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_id = 0
    validation_start_index = train_window_days + purge_days

    while validation_start_index < len(dates):
        validation_end_index = validation_start_index + validation_window_days - 1
        if validation_end_index >= len(dates):
            break

        train_end_index = validation_start_index - purge_days
        train_start_index = train_end_index - train_window_days
        train_dates = dates[train_start_index:train_end_index]
        validation_dates = dates[validation_start_index : validation_end_index + 1]

        rows.append(
            _split_row(
                split_id=split_id,
                train_dates=train_dates,
                validation_dates=validation_dates,
                data=data,
                date_column=date_column,
                purge_days=purge_days,
            )
        )
        split_id += 1
        validation_start_index += step_days

    return pd.DataFrame(rows, columns=_split_metadata_columns())


def _generate_yearly_splits(
    *,
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    dates: pd.DatetimeIndex,
    train_window_days: int,
    date_column: str,
    purge_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_id = 0

    for year in sorted(pd.Series(dates.year).unique()):
        year_dates = dates[dates.year == year]
        if year_dates.empty:
            continue

        validation_start_index = dates.get_loc(year_dates[0])
        if validation_start_index < train_window_days + purge_days:
            continue

        train_end_index = validation_start_index - purge_days
        train_dates = dates[train_end_index - train_window_days : train_end_index]
        rows.append(
            _split_row(
                split_id=split_id,
                train_dates=train_dates,
                validation_dates=year_dates,
                data=data,
                date_column=date_column,
                purge_days=purge_days,
            )
        )
        split_id += 1

    return pd.DataFrame(rows, columns=_split_metadata_columns())


def _split_row(
    *,
    split_id: int,
    train_dates: pd.DatetimeIndex,
    validation_dates: pd.DatetimeIndex,
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    date_column: str,
    purge_days: int,
) -> dict[str, object]:
    train_start = train_dates.min()
    train_end = train_dates.max()
    validation_start = validation_dates.min()
    validation_end = validation_dates.max()

    if train_end >= validation_start:
        raise ValueError("Training dates must occur before validation dates.")

    return {
        "split_id": split_id,
        "train_start_date": train_start,
        "train_end_date": train_end,
        "validation_start_date": validation_start,
        "validation_end_date": validation_end,
        "train_days": len(train_dates),
        "validation_days": len(validation_dates),
        "train_rows": _count_rows_between(data, train_start, train_end, date_column),
        "validation_rows": _count_rows_between(
            data,
            validation_start,
            validation_end,
            date_column,
        ),
        "purge_days": purge_days,
    }


def _unique_dates(
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    *,
    date_column: str,
) -> pd.DatetimeIndex:
    if isinstance(data, pd.DataFrame):
        if date_column not in data.columns:
            raise ValueError(f"Data is missing date column: {date_column}")
        raw_dates = data[date_column]
    elif isinstance(data, pd.Series):
        raw_dates = data
    else:
        raw_dates = data

    dates = pd.to_datetime(raw_dates, errors="coerce")
    dates = pd.DatetimeIndex(dates).dropna().normalize().unique().sort_values()

    if len(dates) == 0:
        raise ValueError("No valid dates were provided.")

    return dates


def _count_rows_between(
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    start: pd.Timestamp,
    end: pd.Timestamp,
    date_column: str,
) -> int:
    if not isinstance(data, pd.DataFrame):
        dates = _unique_dates(data, date_column=date_column)
        return int(((dates >= start) & (dates <= end)).sum())

    dates = pd.to_datetime(data[date_column], errors="coerce").dt.normalize()
    return int(((dates >= start) & (dates <= end)).sum())


def _validate_positive_window(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _empty_split_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=_split_metadata_columns())


def _split_metadata_columns() -> list[str]:
    return [
        "split_id",
        "train_start_date",
        "train_end_date",
        "validation_start_date",
        "validation_end_date",
        "train_days",
        "validation_days",
        "train_rows",
        "validation_rows",
        "purge_days",
    ]


def _attach_locked_test_metadata(
    splits: pd.DataFrame, manifest: dict[str, object] | None
) -> pd.DataFrame:
    if manifest is None:
        return splits
    result = splits.copy()
    result["protocol_version"] = PROTOCOL_VERSION
    for column in [
        "development_end_date",
        "locked_test_start_date",
        "locked_test_end_date",
        "locked_test_days",
        "embargo_days",
        "locked_test_accessed",
    ]:
        result[column] = manifest[column]
    return result
