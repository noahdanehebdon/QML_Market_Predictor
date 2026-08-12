"""Persistence and human-readable reporting for model comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.backtest.portfolio import TRADING_DAYS_PER_YEAR
from market_qml.qml.comparison_types import ComparisonResult


def save_comparison_result(
    result: ComparisonResult, output_dir: str | Path
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "predictions",
        "split_metrics",
        "aggregate_metrics",
        "resource_usage",
        "qsvm_tuning_trials",
        "qsvm_selected_configs",
        "vqc_tuning_trials",
        "vqc_selected_configs",
        "qcnn_tuning_trials",
        "qcnn_selected_configs",
        "sample_manifest",
        "ranking_metrics",
        "portfolio_returns",
        "portfolio_metrics",
        "paired_comparisons",
        "date_block_metrics",
    )
    paths = {}
    for name in names:
        paths[name] = output / f"{name}.parquet"
        getattr(result, name).to_parquet(paths[name], index=False)
    paths["report"] = output / "comparison_report.md"
    paths["report"].write_text(render_comparison_report(result), encoding="utf-8")
    return paths


def render_comparison_report(result: ComparisonResult) -> str:
    pivot = result.aggregate_metrics.pivot(
        index="model_name", columns="metric", values="mean"
    )
    ranked = pivot.sort_values("roc_auc", ascending=False)
    best = ranked.index[0]
    best_qml = pivot.loc[["vqc", "qcnn", "qsvm", "qsvm_tuned"], "roc_auc"].idxmax()
    best_classical = pivot.loc[
        ["logistic_regression", "gradient_boosting"], "roc_auc"
    ].idxmax()
    qml_auc = float(pivot.loc[best_qml, "roc_auc"])
    classical_auc = float(pivot.loc[best_classical, "roc_auc"])
    if classical_auc > qml_auc + 0.02:
        decision = f"QML underperforms the requested classical baselines on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f})."
    elif qml_auc > classical_auc + 0.02:
        decision = f"QML outperforms the requested classical baselines on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f}); confirm the result on additional chronological splits."
    else:
        decision = f"QML and the requested classical baselines behave similarly on mean ROC-AUC ({qml_auc:.4f} versus {classical_auc:.4f}); ranking and portfolio metrics should drive interpretation."
    ranking = result.ranking_metrics.query("scope == 'overall'").set_index("model_name")
    portfolio = result.portfolio_metrics.query("scope == 'overall'").set_index(
        "model_name"
    )
    display = (
        ranked[["accuracy", "roc_auc", "log_loss", "brier_score"]]
        .join(ranking[["rank_information_coefficient", "long_short_spread"]])
        .join(
            portfolio[
                ["cumulative_net_return", "cumulative_net_excess_return", "net_sharpe"]
            ]
        )
    )
    headers = ["model", *display.columns]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend(
        "| " + " | ".join([str(index), *[_format(value) for value in row]]) + " |"
        for index, row in display.iterrows()
    )
    return "\n".join(
        [
            "# QML model comparison",
            "",
            "All models used identical outer validation rows. VQC, QCNN, and QSVM choices were made only from an inner chronological portion of each training window.",
            "",
            f"Portfolio assumptions: {config_text(result.portfolio_returns)}.",
            "",
            *table,
            "",
            f"Classification leader (mean ROC-AUC): **{best}**. Ranking leader (overall rank IC): **{_leader(display['rank_information_coefficient'])}**. Portfolio leader (overall net Sharpe): **{_leader(display['net_sharpe'])}**.",
            "",
            f"Best QML: **{best_qml}**; best requested classical baseline: **{best_classical}**.",
            "",
            f"Decision: {decision}",
            "",
            "Classification uncertainty is recorded in `aggregate_metrics.parquet`; paired bootstrap intervals, sign-permutation tests, Holm correction, effect sizes, and the practical decision threshold are recorded in `paired_comparisons.parquet`. Date-level ranking results are in `ranking_metrics.parquet`; transaction-cost-aware returns and risk metrics are in `portfolio_returns.parquet` and `portfolio_metrics.parquet`.",
            "",
            "Runtime, peak traced memory, selected configurations, tuning trials, and exact sampled-row hashes are retained beside this report.",
        ]
    )


def config_text(portfolio_returns: pd.DataFrame) -> str:
    row = portfolio_returns.iloc[0]
    periods = TRADING_DAYS_PER_YEAR / float(row["rebalance_frequency"])
    return f"{int(row['return_horizon_days'])}-trading-day returns, rebalance every {int(row['rebalance_frequency'])} prediction dates, {periods:g} periods/year, and {float(row['transaction_cost_bps']):g} bps one-way costs"


def _leader(values: pd.Series) -> str:
    available = pd.to_numeric(values, errors="coerce").dropna()
    if available.empty:
        return "not available"
    maximum = available.max()
    return ", ".join(
        str(name) for name, value in available.items() if np.isclose(value, maximum)
    )


def _format(value) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"
