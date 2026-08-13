"""Permutation evidence and fail-closed development research promotion gates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_permutation_evidence(
    predictions: pd.DataFrame,
    *,
    iterations: int = 200,
    random_state: int = 42,
    block_days: int = 1,
) -> pd.DataFrame:
    """Test mean daily rank IC with block sign permutations and Holm correction."""
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if block_days <= 0:
        raise ValueError("block_days must be positive.")
    rng = np.random.default_rng(random_state)
    rows = []
    for model, frame in predictions.groupby("model_name", sort=True):
        groups = [group for _, group in frame.groupby("date", sort=True)]
        observed_values = [
            _rank_ic(group.y_score, group.forward_excess_return) for group in groups
        ]
        observed = _finite_mean(observed_values)
        if not np.isfinite(observed):
            rows.append(
                {
                    "model_name": model,
                    "observed_mean_daily_rank_ic": np.nan,
                    "null_mean": np.nan,
                    "null_std": np.nan,
                    "empirical_p_value": np.nan,
                    "iterations": iterations,
                }
            )
            continue
        daily = np.asarray(observed_values, dtype=float)
        daily = daily[np.isfinite(daily)]
        blocks = np.asarray(
            [
                _finite_mean(daily[start : start + block_days])
                for start in range(0, len(daily), block_days)
            ]
        )
        signs = rng.choice((-1.0, 1.0), size=(iterations, len(blocks)))
        null = (signs * blocks).mean(axis=1)
        rows.append(
            {
                "model_name": model,
                "observed_mean_daily_rank_ic": observed,
                "null_mean": _finite_mean(null),
                "null_std": _finite_std(null),
                "empirical_p_value": float(
                    (1 + np.sum(null >= observed)) / (iterations + 1)
                ),
                "iterations": iterations,
                "block_days": block_days,
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = _holm_adjust(result["empirical_p_value"])
    return result


def build_research_promotion_table(
    ranking_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    baseline_evidence: pd.DataFrame,
    permutation_evidence: pd.DataFrame,
    *,
    minimum_rank_ic: float = 0.02,
    minimum_positive_split_share: float = 0.6,
    maximum_p_value: float = 0.05,
    maximum_turnover: float = 0.75,
    maximum_drawdown: float = 0.3,
    minimum_validation_splits: int = 5,
) -> pd.DataFrame:
    """Require concordant development evidence before locked-test eligibility."""
    overall_rank = ranking_metrics.loc[
        ranking_metrics.scope.eq("overall"),
        ["model_name", "rank_information_coefficient"],
    ]
    split_rank = ranking_metrics.loc[ranking_metrics.scope.eq("split")]
    positive = (
        split_rank.assign(_positive=split_rank.rank_information_coefficient.gt(0))
        .groupby("model_name", as_index=False)
        .agg(
            positive_split_share=("_positive", "mean"),
            validation_splits=("split_id", "nunique"),
        )
    )
    portfolio = portfolio_metrics.loc[
        portfolio_metrics.scope.eq("overall"),
        [
            "model_name",
            "cumulative_net_excess_return",
            "net_max_drawdown",
            "average_turnover",
            "plausibility_status",
            "neutralization",
        ],
    ]
    naive = baseline_evidence[["model_name", "beats_naive"]].copy()
    result = (
        overall_rank.merge(positive, on="model_name", how="left")
        .merge(portfolio, on="model_name", how="left")
        .merge(naive, on="model_name", how="left")
        .merge(
            permutation_evidence[
                ["model_name", "empirical_p_value", "holm_adjusted_p_value"]
            ],
            on="model_name",
            how="left",
        )
    )
    result["passes_rank_ic"] = result.rank_information_coefficient.ge(minimum_rank_ic)
    result["passes_stability"] = result.positive_split_share.ge(
        minimum_positive_split_share
    )
    result["passes_split_count"] = result.validation_splits.ge(
        minimum_validation_splits
    )
    result["passes_permutation"] = result.holm_adjusted_p_value.le(maximum_p_value)
    result["passes_naive"] = result.beats_naive.eq(True)
    result["passes_economics"] = (
        result.cumulative_net_excess_return.gt(0)
        & result.average_turnover.le(maximum_turnover)
        & result.net_max_drawdown.ge(-maximum_drawdown)
        & result.plausibility_status.eq("passed")
        & result.neutralization.eq("sector_equal_weight")
    )
    gates = [
        "passes_rank_ic",
        "passes_stability",
        "passes_split_count",
        "passes_permutation",
        "passes_naive",
        "passes_economics",
    ]
    result["eligible_for_locked_test"] = result[gates].all(axis=1)
    result["decision"] = np.where(
        result.eligible_for_locked_test,
        "eligible_to_freeze_for_locked_test",
        "remain_in_development",
    )
    return result.sort_values(
        ["eligible_for_locked_test", "rank_information_coefficient"],
        ascending=[False, False],
    )


def _holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        result.loc[index] = running
    return result


def _rank_ic(scores, returns) -> float:
    scores = pd.Series(scores).rank(method="average").to_numpy(float)
    returns = pd.Series(returns).rank(method="average").to_numpy(float)
    if len(scores) < 2 or np.std(scores) == 0 or np.std(returns) == 0:
        return np.nan
    return float(np.corrcoef(scores, returns)[0, 1])


def _finite_mean(values) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else np.nan


def _finite_std(values) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
