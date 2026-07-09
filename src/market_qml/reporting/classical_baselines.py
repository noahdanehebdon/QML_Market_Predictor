"""Classical baseline comparison reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_BASELINES = [
    "logistic_regression",
    "ridge_regression",
    "elastic_net",
    "huber_regression",
    "random_forest",
    "random_forest_regressor",
    "gradient_boosting",
    "gradient_boosting_regressor",
]

MODEL_METADATA = {
    "logistic_regression": {
        "model_family": "linear",
        "prediction_task": "classification",
        "target_name": "outperform_spy_5d",
        "score_used_for_ranking": "outperformance_probability",
    },
    "random_forest": {
        "model_family": "tree_ensemble",
        "prediction_task": "classification",
        "target_name": "outperform_spy_5d",
        "score_used_for_ranking": "outperformance_probability",
    },
    "gradient_boosting": {
        "model_family": "tree_ensemble",
        "prediction_task": "classification",
        "target_name": "outperform_spy_5d",
        "score_used_for_ranking": "outperformance_probability",
    },
    "ridge_regression": {
        "model_family": "linear",
        "prediction_task": "regression_ranking",
        "target_name": "forward_excess_return_5d",
        "score_used_for_ranking": "predicted_forward_excess_return",
    },
    "elastic_net": {
        "model_family": "linear",
        "prediction_task": "regression_ranking",
        "target_name": "forward_excess_return_5d",
        "score_used_for_ranking": "predicted_forward_excess_return",
    },
    "huber_regression": {
        "model_family": "linear_robust",
        "prediction_task": "regression_ranking",
        "target_name": "forward_excess_return_5d",
        "score_used_for_ranking": "predicted_forward_excess_return",
    },
    "random_forest_regressor": {
        "model_family": "tree_ensemble",
        "prediction_task": "regression_ranking",
        "target_name": "forward_excess_return_5d",
        "score_used_for_ranking": "predicted_forward_excess_return",
    },
    "gradient_boosting_regressor": {
        "model_family": "tree_ensemble",
        "prediction_task": "regression_ranking",
        "target_name": "forward_excess_return_5d",
        "score_used_for_ranking": "predicted_forward_excess_return",
    },
}

REPORT_COLUMNS = [
    "model_name",
    "model_family",
    "prediction_task",
    "target_name",
    "score_used_for_ranking",
    "classification_roc_auc",
    "classification_average_precision",
    "ranking_long_short_spread",
    "ranking_rank_information_coefficient",
    "portfolio_cumulative_net_return",
    "portfolio_cumulative_net_excess_return",
    "portfolio_net_sharpe",
    "portfolio_net_max_drawdown",
    "composite_rank",
]


def build_classical_baseline_comparison(
    *,
    classification_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    portfolio_risk_metrics: pd.DataFrame,
    expected_models: list[str] | None = None,
) -> pd.DataFrame:
    """Build one comparison table across classical baseline outputs."""
    expected_models = expected_models or EXPECTED_BASELINES
    models = sorted(
        set(expected_models)
        | set(_overall_models(classification_metrics))
        | set(_overall_models(ranking_metrics))
        | set(_overall_models(portfolio_risk_metrics))
    )
    result = pd.DataFrame({"model_name": models})
    result = result.merge(_model_metadata_frame(models), on="model_name", how="left")

    result = result.merge(
        _classification_summary(classification_metrics),
        on="model_name",
        how="left",
    )
    result = result.merge(
        _ranking_summary(ranking_metrics),
        on="model_name",
        how="left",
    )
    result = result.merge(
        _portfolio_summary(portfolio_risk_metrics),
        on="model_name",
        how="left",
    )
    result["composite_rank"] = _composite_rank(result)

    return result[REPORT_COLUMNS].sort_values(
        ["composite_rank", "model_name"],
        na_position="last",
    ).reset_index(drop=True)


def strongest_baseline(comparison: pd.DataFrame) -> str | None:
    """Return the highest-ranked available baseline model."""
    available = comparison.dropna(subset=["composite_rank"])
    if available.empty:
        return None
    return str(available.sort_values(["composite_rank", "model_name"]).iloc[0]["model_name"])


def render_classical_baseline_report(
    comparison: pd.DataFrame,
    *,
    expected_models: list[str] | None = None,
) -> str:
    """Render a Markdown report for the classical baseline comparison."""
    expected_models = expected_models or EXPECTED_BASELINES
    strongest = strongest_baseline(comparison)
    available_models = set(comparison.dropna(subset=["composite_rank"])["model_name"])
    missing_models = [model for model in expected_models if model not in available_models]

    lines = [
        "# Classical Baseline Comparison",
        "",
        f"Strongest available baseline: **{strongest or 'not available'}**.",
        "",
    ]
    if missing_models:
        lines.extend(
            [
                "Missing expected baselines from the available report inputs: "
                + ", ".join(missing_models)
                + ".",
                "",
            ]
        )

    lines.extend(
        [
            "Ranking rule: lower composite rank is better. The composite rank averages "
            "available applicable ranks across classification, ranking, and portfolio "
            "metrics.",
            "",
            "Classification metrics apply to classifier baselines only. Regression/ranking "
            "baselines predict forward excess return, so classifier-only metrics are shown "
            "as `NA` instead of being treated as failures.",
            "",
            _markdown_table(comparison),
            "",
        ]
    )
    return "\n".join(lines)


def save_classical_baseline_report(
    *,
    comparison: pd.DataFrame,
    markdown: str,
    comparison_output: str | Path,
    markdown_output: str | Path,
) -> None:
    """Save comparison table and Markdown report."""
    comparison_output = Path(comparison_output)
    markdown_output = Path(markdown_output)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)

    comparison.to_parquet(comparison_output, index=False)
    markdown_output.write_text(markdown, encoding="utf-8")


def _classification_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = _overall(metrics)
    columns = ["model_name", "roc_auc", "average_precision"]
    if overall.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "classification_roc_auc",
                "classification_average_precision",
            ]
        )
    return overall[columns].rename(
        columns={
            "roc_auc": "classification_roc_auc",
            "average_precision": "classification_average_precision",
        }
    )


def _model_metadata_frame(model_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        metadata = MODEL_METADATA.get(
            model_name,
            {
                "model_family": "unknown",
                "prediction_task": "unknown",
                "target_name": "unknown",
                "score_used_for_ranking": "unknown",
            },
        )
        rows.append({"model_name": model_name, **metadata})
    return pd.DataFrame(rows)


def _ranking_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = _overall(metrics)
    columns = [
        "model_name",
        "long_short_spread",
        "rank_information_coefficient",
    ]
    if overall.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "ranking_long_short_spread",
                "ranking_rank_information_coefficient",
            ]
        )
    return overall[columns].rename(
        columns={
            "long_short_spread": "ranking_long_short_spread",
            "rank_information_coefficient": "ranking_rank_information_coefficient",
        }
    )


def _portfolio_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = _overall(metrics)
    columns = [
        "model_name",
        "cumulative_net_return",
        "cumulative_net_excess_return",
        "net_sharpe",
        "net_max_drawdown",
    ]
    if overall.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "portfolio_cumulative_net_return",
                "portfolio_cumulative_net_excess_return",
                "portfolio_net_sharpe",
                "portfolio_net_max_drawdown",
            ]
        )
    return overall[columns].rename(
        columns={
            "cumulative_net_return": "portfolio_cumulative_net_return",
            "cumulative_net_excess_return": "portfolio_cumulative_net_excess_return",
            "net_sharpe": "portfolio_net_sharpe",
            "net_max_drawdown": "portfolio_net_max_drawdown",
        }
    )


def _overall(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "scope" not in metrics.columns:
        return pd.DataFrame()
    return metrics[metrics["scope"] == "overall"].copy()


def _overall_models(metrics: pd.DataFrame) -> list[str]:
    overall = _overall(metrics)
    if overall.empty or "model_name" not in overall.columns:
        return []
    return overall["model_name"].astype(str).tolist()


def _composite_rank(comparison: pd.DataFrame) -> pd.Series:
    rank_inputs = pd.DataFrame(index=comparison.index)
    shared_higher_is_better = [
        "ranking_long_short_spread",
        "ranking_rank_information_coefficient",
        "portfolio_cumulative_net_return",
        "portfolio_cumulative_net_excess_return",
        "portfolio_net_sharpe",
    ]
    classification_higher_is_better = [
        "classification_roc_auc",
        "classification_average_precision",
    ]
    higher_is_better = shared_higher_is_better + classification_higher_is_better
    for column in higher_is_better:
        rank_inputs[column] = comparison[column].rank(ascending=False, method="min")

    rank_inputs["portfolio_net_max_drawdown"] = comparison[
        "portfolio_net_max_drawdown"
    ].rank(ascending=False, method="min")
    return rank_inputs.mean(axis=1, skipna=True)


def _markdown_table(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    rows = [[_format_value(value) for value in row] for row in data.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_value(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
