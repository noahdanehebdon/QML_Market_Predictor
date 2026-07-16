"""Unified classical and quantum model comparison reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLASSICAL_MODELS = {
    "gradient_boosting",
    "linear_svm",
    "logistic_regression",
    "rbf_svm",
}
QML_MODELS = {"qcnn", "qsvm", "qsvm_tuned", "vqc"}
EXPECTED_MODELS = sorted(CLASSICAL_MODELS | QML_MODELS)

SUMMARY_COLUMNS = [
    "model_name",
    "model_family",
    "classification_accuracy",
    "classification_roc_auc",
    "classification_roc_auc_ci_lower",
    "classification_roc_auc_ci_upper",
    "classification_log_loss",
    "classification_brier_score",
    "ranking_rank_information_coefficient",
    "ranking_long_short_spread",
    "portfolio_cumulative_net_return",
    "portfolio_cumulative_net_excess_return",
    "portfolio_net_sharpe",
    "portfolio_net_max_drawdown",
    "composite_rank",
]


@dataclass(frozen=True)
class ModelComparisonReport:
    """Comparison table, regime leaders, and rendered interpretation."""

    summary: pd.DataFrame
    regime_leaders: pd.DataFrame
    markdown: str
    strongest_model: str | None


def build_model_comparison_report(
    *,
    aggregate_metrics: pd.DataFrame,
    ranking_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    expected_models: list[str] | None = None,
) -> ModelComparisonReport:
    """Combine aligned comparison artifacts into one auditable report."""
    expected_models = expected_models or EXPECTED_MODELS
    classification = _classification_summary(aggregate_metrics)
    ranking = _overall_metrics(
        ranking_metrics,
        {
            "rank_information_coefficient": (
                "ranking_rank_information_coefficient"
            ),
            "long_short_spread": "ranking_long_short_spread",
        },
    )
    portfolio = _overall_metrics(
        portfolio_metrics,
        {
            "cumulative_net_return": "portfolio_cumulative_net_return",
            "cumulative_net_excess_return": (
                "portfolio_cumulative_net_excess_return"
            ),
            "net_sharpe": "portfolio_net_sharpe",
            "net_max_drawdown": "portfolio_net_max_drawdown",
        },
    )

    available_models = set(expected_models)
    for table in (classification, ranking, portfolio):
        if "model_name" in table:
            available_models.update(table["model_name"].astype(str))

    summary = pd.DataFrame({"model_name": sorted(available_models)})
    summary["model_family"] = summary["model_name"].map(_model_family)
    for table in (classification, ranking, portfolio):
        summary = summary.merge(table, on="model_name", how="left")
    summary["composite_rank"] = _composite_rank(summary)
    summary = summary.reindex(columns=SUMMARY_COLUMNS).sort_values(
        ["composite_rank", "model_name"], na_position="last"
    ).reset_index(drop=True)

    strongest = _strongest_model(summary)
    regime_leaders = build_regime_leaders(regime_metrics)
    markdown = render_model_comparison_report(
        summary,
        regime_leaders=regime_leaders,
        strongest_model=strongest,
        expected_models=expected_models,
    )
    return ModelComparisonReport(summary, regime_leaders, markdown, strongest)


def build_regime_leaders(regime_metrics: pd.DataFrame) -> pd.DataFrame:
    """Identify valid classification and ranking leaders in each regime slice."""
    columns = [
        "regime_type",
        "regime",
        "rows",
        "classification_leader",
        "classification_roc_auc",
        "ranking_leader",
        "ranking_rank_information_coefficient",
    ]
    if regime_metrics.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "regime_type",
        "regime",
        "model_name",
        "rows",
        "meets_minimum_rows",
        "roc_auc",
        "rank_information_coefficient",
    }
    _require_columns(regime_metrics, required, "Regime metrics")
    valid = regime_metrics.loc[regime_metrics["meets_minimum_rows"].astype(bool)]
    rows = []
    for (regime_type, regime), group in valid.groupby(
        ["regime_type", "regime"], sort=True
    ):
        classification = _leader(group, "roc_auc")
        ranking = _leader(group, "rank_information_coefficient")
        rows.append(
            {
                "regime_type": regime_type,
                "regime": regime,
                "rows": int(group["rows"].max()),
                "classification_leader": classification[0],
                "classification_roc_auc": classification[1],
                "ranking_leader": ranking[0],
                "ranking_rank_information_coefficient": ranking[1],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def render_model_comparison_report(
    summary: pd.DataFrame,
    *,
    regime_leaders: pd.DataFrame,
    strongest_model: str | None,
    expected_models: list[str],
) -> str:
    """Render a transparent Markdown comparison and limitations section."""
    available = set(summary.dropna(subset=["composite_rank"])["model_name"])
    missing = sorted(set(expected_models) - available)
    best_classical = _family_leader(summary, "classical")
    best_qml = _family_leader(summary, "qml")

    lines = [
        "# Classical and QML Model Comparison",
        "",
        "All reported models use the controlled comparison's identical chronological "
        "validation rows and selected inputs.",
        "",
        f"**Strongest model by composite rank:** **{strongest_model or 'not available'}**.",
        f"**Strongest classical model:** **{best_classical or 'not available'}**.",
        f"**Strongest QML model:** **{best_qml or 'not available'}**.",
        "",
        "The composite is an equal-weight mean of available ordinal ranks across "
        "classification, ranking, and transaction-cost-aware portfolio metrics. It is a "
        "summary aid, not evidence of statistical or economic superiority.",
        "",
    ]
    if missing:
        lines.extend(
            [
                "Expected models without complete comparable outputs: "
                + ", ".join(missing)
                + ".",
                "",
            ]
        )

    lines.extend(["## Overall Metrics", "", _markdown_table(summary), ""])
    lines.extend(["## Regime-Specific Leaders", ""])
    if regime_leaders.empty:
        lines.extend(
            ["No regime slices met the configured minimum sample requirement.", ""]
        )
    else:
        lines.extend([_markdown_table(regime_leaders), ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            _interpretation(summary, best_classical, best_qml),
            "",
            "## Limitations",
            "",
            "- Results come from a reduced, balanced research sample rather than live "
            "capital deployment.",
            "- Confidence intervals use chronological-split bootstrap summaries; overlapping "
            "intervals weaken claims based on small mean differences.",
            "- Regime comparisons are descriptive, involve multiple slices, and may not "
            "persist out of sample.",
            "- Portfolio results depend on the selected universe, rebalance schedule, and "
            "transaction-cost assumptions.",
            "- QML results use exact local simulation and do not include hardware noise, "
            "queueing, or execution costs.",
            "- Composite rank is heuristic and cannot establish quantum advantage or future "
            "trading performance.",
            "",
        ]
    )
    return "\n".join(lines)


def save_model_comparison_report(
    report: ModelComparisonReport,
    *,
    markdown_path: str | Path,
    summary_path: str | Path,
    regime_path: str | Path,
) -> None:
    """Save narrative and machine-readable comparison tables."""
    markdown_path = Path(markdown_path)
    summary_path = Path(summary_path)
    regime_path = Path(regime_path)
    for path in (markdown_path, summary_path, regime_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(report.markdown, encoding="utf-8")
    report.summary.to_csv(summary_path, index=False)
    report.regime_leaders.to_csv(regime_path, index=False)


def _classification_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "metric", "mean", "ci_lower", "ci_upper"}
    _require_columns(metrics, required, "Aggregate classification metrics")
    if metrics.empty:
        return pd.DataFrame(columns=["model_name"])

    means = metrics.pivot_table(
        index="model_name", columns="metric", values="mean", aggfunc="first"
    )
    result = means.reindex(columns=["accuracy", "roc_auc", "log_loss", "brier_score"])
    result.columns = [
        "classification_accuracy",
        "classification_roc_auc",
        "classification_log_loss",
        "classification_brier_score",
    ]
    auc = metrics.loc[metrics["metric"].eq("roc_auc")].set_index("model_name")
    result["classification_roc_auc_ci_lower"] = auc["ci_lower"]
    result["classification_roc_auc_ci_upper"] = auc["ci_upper"]
    return result.reset_index()


def _overall_metrics(metrics: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    required = {"model_name", "scope", *column_map}
    _require_columns(metrics, required, "Overall metrics")
    overall = metrics.loc[metrics["scope"].eq("overall")]
    if overall.empty:
        return pd.DataFrame(columns=["model_name", *column_map.values()])
    return overall[["model_name", *column_map]].rename(columns=column_map)


def _composite_rank(summary: pd.DataFrame) -> pd.Series:
    rank_inputs = pd.DataFrame(index=summary.index)
    higher_is_better = [
        "classification_accuracy",
        "classification_roc_auc",
        "ranking_rank_information_coefficient",
        "ranking_long_short_spread",
        "portfolio_cumulative_net_return",
        "portfolio_cumulative_net_excess_return",
        "portfolio_net_sharpe",
        "portfolio_net_max_drawdown",
    ]
    lower_is_better = ["classification_log_loss", "classification_brier_score"]
    for column in higher_is_better:
        rank_inputs[column] = pd.to_numeric(summary[column], errors="coerce").rank(
            ascending=False, method="min"
        )
    for column in lower_is_better:
        rank_inputs[column] = pd.to_numeric(summary[column], errors="coerce").rank(
            ascending=True, method="min"
        )
    return rank_inputs.mean(axis=1, skipna=True)


def _strongest_model(summary: pd.DataFrame) -> str | None:
    available = summary.dropna(subset=["composite_rank"])
    if available.empty:
        return None
    return str(available.iloc[0]["model_name"])


def _family_leader(summary: pd.DataFrame, family: str) -> str | None:
    available = summary.loc[summary["model_family"].eq(family)].dropna(
        subset=["composite_rank"]
    )
    if available.empty:
        return None
    return str(available.sort_values(["composite_rank", "model_name"]).iloc[0].model_name)


def _model_family(model_name: str) -> str:
    if model_name in QML_MODELS:
        return "qml"
    if model_name in CLASSICAL_MODELS:
        return "classical"
    return "unknown"


def _leader(group: pd.DataFrame, metric: str) -> tuple[str, float]:
    available = group.dropna(subset=[metric]).sort_values(
        [metric, "model_name"], ascending=[False, True]
    )
    if available.empty:
        return "not available", np.nan
    row = available.iloc[0]
    return str(row["model_name"]), float(row[metric])


def _interpretation(
    summary: pd.DataFrame,
    best_classical: str | None,
    best_qml: str | None,
) -> str:
    if best_classical is None or best_qml is None:
        return "Comparable classical and QML results are not both available."
    indexed = summary.set_index("model_name")
    classical_auc = indexed.at[best_classical, "classification_roc_auc"]
    qml_auc = indexed.at[best_qml, "classification_roc_auc"]
    if pd.isna(classical_auc) or pd.isna(qml_auc):
        return "ROC-AUC is unavailable for one or both family leaders."
    difference = float(qml_auc - classical_auc)
    if difference > 0.02:
        return (
            f"{best_qml} leads {best_classical} in mean ROC-AUC by {difference:.4f}, "
            "but this requires confirmation across additional chronological splits."
        )
    if difference < -0.02:
        return (
            f"{best_classical} leads {best_qml} in mean ROC-AUC by "
            f"{abs(difference):.4f}; the current comparison does not show a QML "
            "classification advantage."
        )
    return (
        f"{best_classical} and {best_qml} are within 0.02 mean ROC-AUC. Ranking, "
        "portfolio behavior, confidence intervals, and regime stability should guide "
        "interpretation rather than declaring a family-wide winner."
    )


def _require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{name} are missing columns: " + ", ".join(sorted(missing)))


def _markdown_table(data: pd.DataFrame) -> str:
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data.itertuples(index=False, name=None):
        values = [_format_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_value(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)
