"""Cross-sectional residual targets for stock-specific ranking research."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["symbol", "date"]


def build_residualized_target(
    labels: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    target_column: str,
    numeric_exposures: tuple[str, ...] = (
        "rolling_beta_20d_vs_spy",
        "realized_vol_20d",
        "avg_dollar_volume_20d",
    ),
    categorical_exposures: tuple[str, ...] = ("sector", "size_bucket"),
    minimum_cross_section: int = 20,
) -> pd.DataFrame:
    """Residualize each date using only same-date exposures and realized targets."""
    required_labels = set(KEYS + [target_column])
    missing = required_labels - set(labels)
    if missing:
        raise ValueError("Labels are missing: " + ", ".join(sorted(missing)))
    available_numeric = [column for column in numeric_exposures if column in exposures]
    available_categorical = [
        column for column in categorical_exposures if column in exposures
    ]
    if not available_numeric and not available_categorical:
        raise ValueError("No requested residualization exposures are available.")
    columns = KEYS + available_numeric + available_categorical
    merged = labels.merge(
        exposures[columns].drop_duplicates(KEYS),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    output_column = f"residualized_{target_column}"
    merged[output_column] = np.nan
    for _, index in merged.groupby("date", sort=False).groups.items():
        frame = merged.loc[index]
        y = pd.to_numeric(frame[target_column], errors="coerce")
        design = pd.DataFrame(index=frame.index)
        for column in available_numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = values.median()
            scale = values.mad() if hasattr(values, "mad") else None
            scale = float((values - median).abs().median() * 1.4826)
            design[column] = (values - median) / (scale if scale > 0 else 1.0)
        if available_categorical:
            categorical = pd.get_dummies(
                frame[available_categorical].fillna("unknown").astype(str),
                prefix=available_categorical,
                dtype=float,
            )
            design = pd.concat([design, categorical], axis=1)
        valid = y.notna() & design.notna().all(axis=1)
        if valid.sum() < max(minimum_cross_section, design.shape[1] + 2):
            continue
        matrix = np.column_stack([np.ones(valid.sum()), design.loc[valid].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(matrix, y.loc[valid].to_numpy(), rcond=None)
        merged.loc[valid[valid].index, output_column] = (
            y.loc[valid].to_numpy() - matrix @ coefficients
        )
    return merged[KEYS + [output_column]].copy()
