"""Numerically explicit statistics for degenerate research samples."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def safe_correlation(
    left: Iterable[object],
    right: Iterable[object],
    *,
    method: str = "pearson",
) -> float:
    """Return a finite correlation or NaN when it is mathematically undefined."""
    pair = (
        pd.DataFrame(
            {
                "left": pd.to_numeric(pd.Series(left), errors="coerce"),
                "right": pd.to_numeric(pd.Series(right), errors="coerce"),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < 2:
        return np.nan
    if method == "spearman":
        pair = pair.rank(method="average")
    elif method != "pearson":
        raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
    left_values = pair["left"].to_numpy(dtype=float)
    right_values = pair["right"].to_numpy(dtype=float)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        return np.nan
    result = float(np.dot(left_centered, right_centered) / denominator)
    return float(np.clip(result, -1.0, 1.0)) if np.isfinite(result) else np.nan


def absolute_correlation_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Return absolute Pearson correlations, using zero for undefined pairs."""
    columns = list(frame.columns)
    result = pd.DataFrame(0.0, index=columns, columns=columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index:]:
            correlation = safe_correlation(frame[left], frame[right])
            value = 0.0 if not np.isfinite(correlation) else abs(correlation)
            result.loc[left, right] = value
            result.loc[right, left] = value
    return result
