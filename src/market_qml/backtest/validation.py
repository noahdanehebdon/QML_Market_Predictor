"""Locked-test partitioning and paired statistical model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROTOCOL_VERSION = "locked-test-v1"


def prediction_date_block_metrics(
    predictions: pd.DataFrame, *, block_days: int = 20
) -> pd.DataFrame:
    """Compute out-of-sample ROC-AUC on non-overlapping chronological date blocks."""
    required = {"model_name", "split_id", "date", "y_true", "y_score"}
    missing = required - set(predictions)
    if missing:
        raise ValueError("Predictions are missing: " + ", ".join(sorted(missing)))
    if block_days <= 0:
        raise ValueError("block_days must be positive.")
    rows = []
    for (model_name, split_id), group in predictions.groupby(
        ["model_name", "split_id"], sort=True
    ):
        ordered = group.assign(date=pd.to_datetime(group["date"])).sort_values("date")
        dates = pd.DatetimeIndex(ordered["date"].unique()).sort_values()
        for block_index, block_dates in enumerate(
            np.array_split(dates, np.ceil(len(dates) / block_days).astype(int))
        ):
            block = ordered.loc[ordered["date"].isin(block_dates)]
            if block["y_true"].nunique() < 2:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "split_id": f"{int(split_id)}-{block_index}",
                    "outer_split_id": int(split_id),
                    "block_id": block_index,
                    "block_start_date": block_dates.min(),
                    "block_end_date": block_dates.max(),
                    "rows": len(block),
                    "roc_auc": roc_auc_score(block["y_true"], block["y_score"]),
                }
            )
    return pd.DataFrame(rows)


def partition_locked_test(
    data: pd.DataFrame,
    *,
    locked_test_days: int,
    embargo_days: int = 0,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Separate development and locked-test rows using unique trading dates."""
    if locked_test_days <= 0:
        raise ValueError("locked_test_days must be positive.")
    if embargo_days < 0:
        raise ValueError("embargo_days cannot be negative.")
    if date_column not in data:
        raise ValueError(f"Data is missing date column: {date_column}")
    dates = pd.DatetimeIndex(
        pd.to_datetime(data[date_column], errors="coerce").dropna().unique()
    ).sort_values()
    if len(dates) <= locked_test_days + embargo_days:
        raise ValueError("Not enough dates to reserve the locked test and embargo.")
    locked_dates = dates[-locked_test_days:]
    development_dates = dates[: -(locked_test_days + embargo_days)]
    normalized = pd.to_datetime(data[date_column], errors="coerce")
    development = data.loc[normalized.isin(development_dates)].copy()
    locked = data.loc[normalized.isin(locked_dates)].copy()
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "development_end_date": development_dates.max(),
        "locked_test_start_date": locked_dates.min(),
        "locked_test_end_date": locked_dates.max(),
        "locked_test_days": locked_test_days,
        "embargo_days": embargo_days,
        "locked_test_accessed": False,
    }
    return development, locked, manifest


def log_locked_test_access(
    manifest: dict[str, object],
    *,
    reason: str,
    audit_path: str | Path,
) -> dict[str, object]:
    """Record a deliberate final-test access before results are inspected."""
    if not reason.strip():
        raise ValueError("A non-empty locked-test access reason is required.")
    record = {
        **manifest,
        "locked_test_accessed": True,
        "access_reason": reason.strip(),
        "accessed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return record


def paired_model_comparisons(
    metrics: pd.DataFrame,
    *,
    metric: str,
    baseline_model: str,
    bootstrap_iterations: int = 2000,
    practical_threshold: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap and sign-permutation paired split differences versus a baseline."""
    required = {"model_name", "split_id", metric}
    missing = required - set(metrics)
    if missing:
        raise ValueError("Metrics are missing: " + ", ".join(sorted(missing)))
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive.")
    pivot = metrics.pivot(index="split_id", columns="model_name", values=metric)
    if baseline_model not in pivot:
        raise ValueError(f"Baseline model is absent: {baseline_model}")
    rng = np.random.default_rng(random_state)
    rows = []
    candidates = [name for name in sorted(pivot.columns) if name != baseline_model]
    for candidate in candidates:
        paired = pivot[[candidate, baseline_model]].dropna()
        differences = (paired[candidate] - paired[baseline_model]).to_numpy(float)
        if not len(differences):
            continue
        boot = np.asarray(
            [
                rng.choice(differences, len(differences), replace=True).mean()
                for _ in range(bootstrap_iterations)
            ]
        )
        signs = rng.choice((-1.0, 1.0), size=(bootstrap_iterations, len(differences)))
        null_means = (signs * differences).mean(axis=1)
        observed = float(differences.mean())
        rows.append(
            {
                "candidate_model": candidate,
                "baseline_model": baseline_model,
                "metric": metric,
                "paired_splits": len(differences),
                "mean_difference": observed,
                "ci_lower": float(np.quantile(boot, 0.025)),
                "ci_upper": float(np.quantile(boot, 0.975)),
                "permutation_p_value": float(
                    (np.count_nonzero(np.abs(null_means) >= abs(observed)) + 1)
                    / (bootstrap_iterations + 1)
                ),
                "practical_threshold": practical_threshold,
                "practically_meaningful": abs(observed) >= practical_threshold,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values("permutation_p_value").reset_index(drop=True)
    count = len(result)
    result["holm_adjusted_p_value"] = np.maximum.accumulate(
        np.minimum(1.0, result["permutation_p_value"] * (count - np.arange(count)))
    )
    result["statistically_significant"] = result["holm_adjusted_p_value"] < 0.05
    result["decision"] = np.where(
        result["statistically_significant"] & result["practically_meaningful"],
        "material_difference",
        "insufficient_evidence",
    )
    return result.sort_values("candidate_model").reset_index(drop=True)
