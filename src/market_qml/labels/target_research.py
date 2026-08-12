"""Development-only diagnostics and selection for prediction targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_qml.backtest.validation import PROTOCOL_VERSION, partition_locked_test


def target_catalog(labels: pd.DataFrame, *, benchmark: str = "SPY") -> pd.DataFrame:
    """Describe target timing and missing-label rules from a multi-horizon table."""
    rows: list[dict[str, object]] = []
    for horizon in sorted(pd.to_numeric(labels["label_horizon_days"]).unique()):
        horizon = int(horizon)
        names = {
            "binary": f"outperform_{benchmark.lower()}_{horizon}d",
            "neutral_zone": f"outperform_{benchmark.lower()}_{horizon}d_neutral",
            "continuous": f"forward_excess_return_{horizon}d",
            "volatility_normalized": f"vol_normalized_excess_return_{horizon}d",
            "cross_sectional_rank": f"cross_sectional_rank_{horizon}d",
            "sector_relative": f"sector_relative_return_{horizon}d",
            "sector_rank": f"sector_relative_rank_{horizon}d",
        }
        for family, name in names.items():
            if name not in labels:
                continue
            rows.append(
                {
                    "target_name": name,
                    "target_family": family,
                    "horizon_trading_days": horizon,
                    "benchmark": benchmark.upper(),
                    "purge_days": horizon,
                    "timing_rule": f"close[t+{horizon}] / close[t] - 1",
                    "missing_label_rule": (
                        "NA when the future price or same-date benchmark is unavailable; "
                        "neutral-zone rows inside the threshold are also NA"
                    ),
                }
            )
    return pd.DataFrame(rows)


def research_target_candidates(
    labels: pd.DataFrame,
    *,
    locked_test_days: int,
    embargo_days: int = 5,
    benchmark: str = "SPY",
    period_frequency: str = "Y",
    inner_folds: int = 3,
    practical_score_margin: float = 0.01,
    minimum_validation_periods: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Measure candidates using development dates only and select two primaries."""
    development, _locked, manifest = partition_locked_test(
        labels,
        locked_test_days=locked_test_days,
        embargo_days=embargo_days,
    )
    catalog = target_catalog(development, benchmark=benchmark)
    rows: list[dict[str, object]] = []
    for candidate in catalog.itertuples(index=False):
        horizon_rows = development.loc[
            development["label_horizon_days"] == candidate.horizon_trading_days
        ].copy()
        horizon_rows["period"] = horizon_rows["date"].dt.to_period(period_frequency)
        for period, group in horizon_rows.groupby("period", sort=True):
            values = pd.to_numeric(group[candidate.target_name], errors="coerce")
            valid = group.loc[values.notna()].copy()
            values = values.dropna()
            if values.empty:
                continue
            binary = candidate.target_family in {"binary", "neutral_zone"}
            turnover = (
                valid.sort_values(["symbol", "date"])
                .groupby("symbol")[candidate.target_name]
                .diff()
                .abs()
                .mean()
            )
            autocorrelation = (
                valid.groupby("symbol")[candidate.target_name]
                .apply(_safe_autocorrelation)
                .mean()
            )
            economic_column = f"forward_excess_return_{candidate.horizon_trading_days}d"
            valid = valid.sort_values(["symbol", "date"])
            valid["_purged_lagged_target"] = valid.groupby("symbol")[
                candidate.target_name
            ].shift(candidate.horizon_trading_days)
            rank_ic = (
                valid.groupby("date")
                .apply(
                    lambda frame: (
                        frame["_purged_lagged_target"].corr(
                            frame[economic_column], method="spearman"
                        )
                        if frame["_purged_lagged_target"].nunique() > 1
                        and frame[economic_column].nunique() > 1
                        else np.nan
                    ),
                    include_groups=False,
                )
                .mean()
            )
            rows.append(
                {
                    "target_name": candidate.target_name,
                    "target_family": candidate.target_family,
                    "horizon_trading_days": candidate.horizon_trading_days,
                    "purge_days": candidate.purge_days,
                    "period": str(period),
                    "rows": len(values),
                    "missing_rate": 1.0 - len(values) / len(group),
                    "positive_rate": float(values.mean()) if binary else np.nan,
                    "turnover": float(turnover) if pd.notna(turnover) else np.nan,
                    "autocorrelation": (
                        float(autocorrelation) if pd.notna(autocorrelation) else np.nan
                    ),
                    "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                    "economic_magnitude": float(
                        pd.to_numeric(valid[economic_column], errors="coerce")
                        .abs()
                        .mean()
                    ),
                }
            )
    diagnostics = _assign_nested_roles(pd.DataFrame(rows), inner_folds=inner_folds)
    selection = _select_primary_targets(
        diagnostics,
        practical_score_margin=practical_score_margin,
        minimum_validation_periods=minimum_validation_periods,
    )
    manifest = {
        **manifest,
        "protocol_version": PROTOCOL_VERSION,
        "selection_scope": "development_only",
        "selection_validation": "nested_chronological_inner_outer",
        "inner_folds": inner_folds,
        "practical_score_margin": practical_score_margin,
        "minimum_validation_periods": minimum_validation_periods,
        "locked_test_rows_inspected": 0,
    }
    return diagnostics, selection, manifest


def _select_primary_targets(
    diagnostics: pd.DataFrame,
    *,
    practical_score_margin: float,
    minimum_validation_periods: int,
) -> pd.DataFrame:
    """Prefer stable, economically meaningful candidates without test access."""
    if diagnostics.empty:
        return pd.DataFrame()
    selection_rows = diagnostics.loc[
        diagnostics["validation_role"].isin(["inner_validation", "outer_validation"])
    ]
    summary = selection_rows.groupby(
        ["target_name", "target_family", "horizon_trading_days", "purge_days"],
        as_index=False,
    ).agg(
        periods=("period", "nunique"),
        mean_rank_ic=("rank_ic", "mean"),
        rank_ic_stability=("rank_ic", "std"),
        mean_economic_magnitude=("economic_magnitude", "mean"),
        mean_missing_rate=("missing_rate", "mean"),
        class_balance=("positive_rate", lambda value: 1 - abs(value.mean() - 0.5) * 2),
        positive_rank_ic_share=("rank_ic", lambda value: (value.dropna() > 0).mean()),
    )
    summary["selection_score"] = (
        summary["mean_rank_ic"].abs().fillna(0)
        - summary["rank_ic_stability"].fillna(0)
        - summary["mean_missing_rate"]
        + summary["class_balance"].fillna(0) * 0.1
    )
    classification = summary[summary["target_family"].isin(["binary", "neutral_zone"])]
    ranking = summary[~summary["target_family"].isin(["binary", "neutral_zone"])]
    selected = []
    for role, candidates in [("classification", classification), ("ranking", ranking)]:
        if not candidates.empty:
            winner = (
                candidates.sort_values(
                    ["selection_score", "horizon_trading_days"], ascending=[False, True]
                )
                .iloc[0]
                .to_dict()
            )
            winner["selected_role"] = role
            baseline_name = (
                "outperform_spy_5d"
                if role == "classification"
                else "forward_excess_return_5d"
            )
            baseline = summary.loc[summary["target_name"] == baseline_name]
            baseline_score = (
                float(baseline.iloc[0]["selection_score"])
                if not baseline.empty
                else np.nan
            )
            winner["comparison_baseline"] = baseline_name
            winner["baseline_score"] = baseline_score
            winner["improves_on_baseline"] = bool(
                pd.isna(baseline_score)
                or winner["selection_score"] >= baseline_score + practical_score_margin
            )
            winner["passes_stability_gates"] = bool(
                winner["periods"] >= minimum_validation_periods
                and winner["mean_missing_rate"] <= 0.2
                and winner["positive_rank_ic_share"] >= 0.5
                and (
                    role != "classification"
                    or pd.isna(winner["class_balance"])
                    or winner["class_balance"] >= 0.6
                )
            )
            winner["decision"] = (
                "promote_candidate"
                if winner["improves_on_baseline"] and winner["passes_stability_gates"]
                else "retain_baseline_null_result"
            )
            selected.append(winner)
    return pd.DataFrame(selected)


def _assign_nested_roles(
    diagnostics: pd.DataFrame, *, inner_folds: int
) -> pd.DataFrame:
    if inner_folds < 2:
        raise ValueError("inner_folds must be at least two.")
    if diagnostics.empty:
        return diagnostics
    result = diagnostics.copy()
    periods = sorted(result["period"].unique())
    result["validation_role"] = "calibration"
    result["fold_id"] = pd.NA
    if periods:
        result.loc[result["period"] == periods[-1], "validation_role"] = (
            "outer_validation"
        )
        result.loc[result["period"] == periods[-1], "fold_id"] = 0
    inner_periods = periods[max(0, len(periods) - inner_folds - 1) : -1]
    for fold_id, period in enumerate(inner_periods):
        mask = result["period"] == period
        result.loc[mask, "validation_role"] = "inner_validation"
        result.loc[mask, "fold_id"] = fold_id
    return result


def _safe_autocorrelation(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if (
        len(numeric) < 3
        or numeric.iloc[:-1].nunique() < 2
        or numeric.iloc[1:].nunique() < 2
    ):
        return np.nan
    return float(numeric.autocorr())
