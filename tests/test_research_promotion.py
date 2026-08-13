import pandas as pd

from market_qml.reporting.research_promotion import (
    build_research_promotion_table,
    rank_ic_permutation_evidence,
)


def test_permutation_evidence_detects_ordered_signal():
    rows = []
    for day in range(8):
        for index in range(20):
            rows.append(
                {
                    "model_name": "ordered",
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "y_score": index,
                    "forward_excess_return": index / 100,
                }
            )
    result = rank_ic_permutation_evidence(
        pd.DataFrame(rows), iterations=99, random_state=7
    )

    assert result.loc[0, "observed_mean_daily_rank_ic"] == 1.0
    assert result.loc[0, "empirical_p_value"] <= 0.02


def test_promotion_fails_closed_when_portfolio_is_implausible():
    ranking = pd.DataFrame(
        {
            "model_name": ["model", "model", "model"],
            "split_id": [0, 1, pd.NA],
            "scope": ["split", "split", "overall"],
            "rank_information_coefficient": [0.05, 0.04, 0.045],
        }
    )
    portfolio = pd.DataFrame(
        {
            "model_name": ["model"],
            "scope": ["overall"],
            "cumulative_net_excess_return": [1.0],
            "net_max_drawdown": [-0.1],
            "average_turnover": [0.2],
            "plausibility_status": ["invalid_extreme_period_return"],
            "neutralization": ["sector_equal_weight"],
        }
    )
    baseline = pd.DataFrame({"model_name": ["model"], "beats_naive": [True]})
    permutation = pd.DataFrame(
        {
            "model_name": ["model"],
            "empirical_p_value": [0.01],
            "holm_adjusted_p_value": [0.01],
        }
    )

    result = build_research_promotion_table(ranking, portfolio, baseline, permutation)

    assert result.loc[0, "eligible_for_locked_test"] == False  # noqa: E712
    assert result.loc[0, "decision"] == "remain_in_development"
