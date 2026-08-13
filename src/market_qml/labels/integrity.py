"""Deterministic return-integrity diagnostics applied before model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnIntegrityRules:
    minimum_return: float = -1.0
    maximum_absolute_return: float = 2.0
    maximum_cross_sectional_robust_z: float = 20.0


def add_return_integrity_flags(
    labels: pd.DataFrame,
    *,
    return_column: str,
    rules: ReturnIntegrityRules = ReturnIntegrityRules(),
) -> pd.DataFrame:
    """Flag impossible and extreme returns without silently clipping outcomes."""
    required = {"symbol", "date", return_column}
    missing = required - set(labels)
    if missing:
        raise ValueError("Labels are missing: " + ", ".join(sorted(missing)))
    if rules.minimum_return < -1 or rules.maximum_absolute_return <= 0:
        raise ValueError("Return-integrity thresholds are invalid.")
    result = labels.copy()
    values = pd.to_numeric(result[return_column], errors="coerce")
    median = values.groupby(result["date"]).transform("median")
    deviation = (values - median).abs()
    mad = deviation.groupby(result["date"]).transform("median")
    robust_z = (values - median) / (1.4826 * mad.replace(0, np.nan))
    result["return_integrity_robust_z"] = robust_z
    result["return_integrity_valid"] = (
        values.notna()
        & values.ge(rules.minimum_return)
        & values.abs().le(rules.maximum_absolute_return)
        & robust_z.abs().fillna(0).le(rules.maximum_cross_sectional_robust_z)
    )
    result["return_integrity_status"] = "passed"
    result.loc[values.lt(rules.minimum_return), "return_integrity_status"] = (
        "impossible_loss"
    )
    result.loc[
        values.abs().gt(rules.maximum_absolute_return), "return_integrity_status"
    ] = "extreme_absolute_return"
    result.loc[
        robust_z.abs().gt(rules.maximum_cross_sectional_robust_z)
        & values.abs().le(rules.maximum_absolute_return),
        "return_integrity_status",
    ] = "extreme_cross_sectional_return"
    result.loc[values.isna(), "return_integrity_status"] = "missing_return"
    return result
