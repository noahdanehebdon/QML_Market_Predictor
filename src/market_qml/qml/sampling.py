"""Reproducible sampling utilities for QML datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from market_qml.qml.pca import DEFAULT_QML_PCA_FEATURE_PATH


DEFAULT_QML_SAMPLE_PATH = Path("data/features/qml_sample.parquet")
DEFAULT_QML_SAMPLE_METADATA_PATH = Path("data/processed/qml_sample_metadata.parquet")
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TRAIN_ROWS_PER_SPLIT = 512
DEFAULT_MAX_VALIDATION_ROWS_PER_SPLIT = 256

KEY_COLUMNS = ["symbol", "date", "split_id", "sample_role"]
REQUIRED_COLUMNS = KEY_COLUMNS + ["target"]


@dataclass(frozen=True)
class QMLSampleResult:
    """Sampled QML rows and auditable sampling metadata."""

    sample: pd.DataFrame
    metadata: pd.DataFrame


def build_qml_sample(
    qml_features: pd.DataFrame,
    *,
    max_train_rows_per_split: int | None = DEFAULT_MAX_TRAIN_ROWS_PER_SPLIT,
    max_validation_rows_per_split: int | None = DEFAULT_MAX_VALIDATION_ROWS_PER_SPLIT,
    max_dates_per_split_role: int | None = None,
    max_symbols: int | None = None,
    balance_classes: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> QMLSampleResult:
    """Build a reproducible, time-ordered sample from QML feature rows."""
    _validate_qml_features(qml_features)
    _validate_optional_positive(max_train_rows_per_split, "max_train_rows_per_split")
    _validate_optional_positive(
        max_validation_rows_per_split,
        "max_validation_rows_per_split",
    )
    _validate_optional_positive(max_dates_per_split_role, "max_dates_per_split_role")
    _validate_optional_positive(max_symbols, "max_symbols")

    data = qml_features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any():
        raise ValueError("QML feature table contains invalid dates.")

    if max_symbols is not None:
        data = _filter_symbols(data, max_symbols=max_symbols, random_seed=random_seed)

    sample_frames = []
    metadata_rows = []
    grouped = data.groupby(["split_id", "sample_role"], sort=True)
    for (split_id, sample_role), group in grouped:
        role_seed = _role_seed(
            random_seed=random_seed,
            split_id=int(split_id),
            sample_role=str(sample_role),
        )
        date_filtered = _limit_dates(
            group,
            max_dates=max_dates_per_split_role,
            random_seed=role_seed,
        )
        working, balanced = _balance_classes(
            date_filtered,
            enabled=balance_classes,
            random_seed=role_seed,
        )
        max_rows = _max_rows_for_role(
            str(sample_role),
            max_train_rows_per_split=max_train_rows_per_split,
            max_validation_rows_per_split=max_validation_rows_per_split,
        )
        sampled = _limit_rows(
            working,
            max_rows=max_rows,
            random_seed=role_seed,
            preserve_binary_balance=balanced,
        )
        sampled = _sort_sample(sampled)
        sample_frames.append(sampled)
        metadata_rows.append(
            _metadata_row(
                original=group,
                after_symbol_filter=group,
                after_date_filter=date_filtered,
                after_balance=working,
                sampled=sampled,
                split_id=int(split_id),
                sample_role=str(sample_role),
                random_seed=random_seed,
                role_seed=role_seed,
                max_rows=max_rows,
                max_dates_per_split_role=max_dates_per_split_role,
                max_symbols=max_symbols,
                balance_requested=balance_classes,
                balanced=balanced,
            )
        )

    if not sample_frames:
        raise ValueError("Sampling produced no rows.")

    return QMLSampleResult(
        sample=pd.concat(sample_frames, ignore_index=True),
        metadata=pd.DataFrame(metadata_rows).sort_values(
            ["split_id", "sample_role"],
        ).reset_index(drop=True),
    )


def load_qml_features(path: str | Path = DEFAULT_QML_PCA_FEATURE_PATH) -> pd.DataFrame:
    """Load QML feature rows from parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QML feature table not found: {path}")
    return pd.read_parquet(path)


def save_qml_sample(
    sample: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_SAMPLE_PATH,
) -> None:
    """Save sampled QML rows to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_path, index=False)


def save_qml_sample_metadata(
    metadata: pd.DataFrame,
    output_path: str | Path = DEFAULT_QML_SAMPLE_METADATA_PATH,
) -> None:
    """Save sampling metadata to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(output_path, index=False)


def _filter_symbols(
    data: pd.DataFrame,
    *,
    max_symbols: int,
    random_seed: int,
) -> pd.DataFrame:
    symbols = sorted(data["symbol"].astype(str).unique())
    if len(symbols) <= max_symbols:
        return data

    selected = (
        pd.Series(symbols)
        .sample(n=max_symbols, random_state=random_seed, replace=False)
        .sort_values()
        .tolist()
    )
    return data[data["symbol"].isin(selected)].copy()


def _limit_dates(
    data: pd.DataFrame,
    *,
    max_dates: int | None,
    random_seed: int,
) -> pd.DataFrame:
    if max_dates is None:
        return data.copy()

    dates = pd.Series(sorted(data["date"].dropna().unique()))
    if len(dates) <= max_dates:
        return data.copy()

    selected_dates = (
        dates.sample(n=max_dates, random_state=random_seed, replace=False)
        .sort_values()
        .tolist()
    )
    return data[data["date"].isin(selected_dates)].copy()


def _balance_classes(
    data: pd.DataFrame,
    *,
    enabled: bool,
    random_seed: int,
) -> tuple[pd.DataFrame, bool]:
    if not enabled:
        return data.copy(), False

    target = pd.to_numeric(data["target"], errors="coerce")
    unique_targets = sorted(target.dropna().unique().tolist())
    if set(unique_targets) != {0, 1}:
        return data.copy(), False

    working = data.assign(_target=target.astype(int))
    counts = working["_target"].value_counts()
    if len(counts) < 2 or counts.min() == 0:
        return data.copy(), False

    class_size = int(counts.min())
    frames = []
    for target_value, class_rows in working.groupby("_target", sort=True):
        frames.append(
            class_rows.sample(
                n=class_size,
                random_state=random_seed + int(target_value),
                replace=False,
            )
        )
    return pd.concat(frames, ignore_index=True).drop(columns=["_target"]), True


def _limit_rows(
    data: pd.DataFrame,
    *,
    max_rows: int | None,
    random_seed: int,
    preserve_binary_balance: bool = False,
) -> pd.DataFrame:
    if max_rows is None or len(data) <= max_rows:
        return data.copy()
    if preserve_binary_balance:
        balanced = _limit_binary_balanced_rows(
            data,
            max_rows=max_rows,
            random_seed=random_seed,
        )
        if balanced is not None:
            return balanced
    return data.sample(n=max_rows, random_state=random_seed, replace=False).copy()


def _limit_binary_balanced_rows(
    data: pd.DataFrame,
    *,
    max_rows: int,
    random_seed: int,
) -> pd.DataFrame | None:
    target = pd.to_numeric(data["target"], errors="coerce")
    unique_targets = sorted(target.dropna().unique().tolist())
    if set(unique_targets) != {0, 1} or max_rows < 2:
        return None

    working = data.assign(_target=target.astype(int))
    counts = working["_target"].value_counts()
    per_class = min(int(counts.min()), max_rows // 2)
    if per_class == 0:
        return None

    frames = []
    for target_value, class_rows in working.groupby("_target", sort=True):
        frames.append(
            class_rows.sample(
                n=per_class,
                random_state=random_seed + int(target_value),
                replace=False,
            )
        )
    return pd.concat(frames, ignore_index=True).drop(columns=["_target"])


def _sort_sample(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.sort_values(
        ["split_id", "sample_role", "date", "symbol"],
    ).reset_index(drop=True)


def _metadata_row(
    *,
    original: pd.DataFrame,
    after_symbol_filter: pd.DataFrame,
    after_date_filter: pd.DataFrame,
    after_balance: pd.DataFrame,
    sampled: pd.DataFrame,
    split_id: int,
    sample_role: str,
    random_seed: int,
    role_seed: int,
    max_rows: int | None,
    max_dates_per_split_role: int | None,
    max_symbols: int | None,
    balance_requested: bool,
    balanced: bool,
) -> dict[str, object]:
    target_counts = (
        sampled["target"].value_counts(dropna=False).sort_index().to_dict()
    )
    return {
        "split_id": split_id,
        "sample_role": sample_role,
        "random_seed": random_seed,
        "role_seed": role_seed,
        "max_rows": max_rows,
        "max_dates_per_split_role": max_dates_per_split_role,
        "max_symbols": max_symbols,
        "balance_requested": balance_requested,
        "balanced_classes": balanced,
        "original_rows": len(original),
        "rows_after_symbol_filter": len(after_symbol_filter),
        "rows_after_date_filter": len(after_date_filter),
        "rows_after_balance": len(after_balance),
        "sampled_rows": len(sampled),
        "unique_symbols": sampled["symbol"].nunique(),
        "unique_dates": sampled["date"].nunique(),
        "target_counts": str(target_counts),
        "start_date": sampled["date"].min() if not sampled.empty else pd.NaT,
        "end_date": sampled["date"].max() if not sampled.empty else pd.NaT,
    }


def _max_rows_for_role(
    sample_role: str,
    *,
    max_train_rows_per_split: int | None,
    max_validation_rows_per_split: int | None,
) -> int | None:
    if sample_role == "train":
        return max_train_rows_per_split
    if sample_role == "validation":
        return max_validation_rows_per_split
    return max_train_rows_per_split


def _role_seed(*, random_seed: int, split_id: int, sample_role: str) -> int:
    role_offset = 0 if sample_role == "train" else 10_000
    return random_seed + split_id * 1_000 + role_offset


def _validate_qml_features(qml_features: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(qml_features.columns)
    if missing:
        raise ValueError(
            "QML feature table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    pca_columns = [column for column in qml_features.columns if column.startswith("pca_")]
    if not pca_columns:
        raise ValueError("QML feature table does not contain PCA component columns.")
    if qml_features.empty:
        raise ValueError("QML feature table is empty.")


def _validate_optional_positive(value: int | None, name: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")
