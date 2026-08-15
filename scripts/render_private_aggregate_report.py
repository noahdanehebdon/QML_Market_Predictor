"""Render aggregate-only experiment metrics from a private R2 report prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TABLE_COLUMNS = {
    "aggregate_metrics": [
        "lane",
        "model_name",
        "model_family",
        "roc_auc",
        "log_loss",
        "brier_score",
        "rank_ic",
    ],
    "paired_comparisons": [
        "lane",
        "baseline_model",
        "candidate_model",
        "metric",
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "decision",
    ],
    "portfolio_summary": [
        "lane",
        "model_name",
        "rows",
        "return_horizon_days",
        "rebalance_frequency",
        "periods_per_year",
        "neutralization",
        "cumulative_net_return",
        "cumulative_net_excess_return",
        "net_sharpe",
        "net_excess_sharpe",
        "net_max_drawdown",
        "minimum_net_return",
        "maximum_net_return",
        "period_returns_over_100pct",
        "plausibility_status",
        "average_turnover",
    ],
    "resource_summary": [
        "lane",
        "model_name",
        "runtime_seconds",
        "peak_memory_mb",
    ],
    "research_promotion": [
        "model_name",
        "rank_information_coefficient",
        "positive_split_share",
        "validation_splits",
        "cumulative_net_excess_return",
        "net_max_drawdown",
        "average_turnover",
        "plausibility_status",
        "neutralization",
        "beats_naive",
        "empirical_p_value",
        "holm_adjusted_p_value",
        "passes_rank_ic",
        "passes_stability",
        "passes_split_count",
        "passes_permutation",
        "passes_naive",
        "passes_economics",
        "eligible_for_locked_test",
        "decision",
    ],
}


def render_report(input_dir: Path) -> str:
    """Render allowlisted aggregate columns without row-level predictions."""
    conclusion_path = input_dir / "conclusion.json"
    sections = ["# Aggregate development results", ""]
    if conclusion_path.exists():
        conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
        sections.extend(
            [
                f"**Decision:** {conclusion['decision']}",
                "",
                f"Locked test accessed: `{conclusion['locked_test_accessed']}`",
                "",
            ]
        )
        if "simulator_winner" in conclusion:
            sections.extend(
                [
                    f"Simulator winner: `{conclusion['simulator_winner']}`",
                    "",
                    f"Hardware candidate: `{conclusion['hardware_candidate']}`",
                    "",
                    f"Qualified for hardware: `{conclusion['qualified_for_hardware']}`",
                    "",
                    "Hardware execution path: "
                    f"`{conclusion['hardware_execution_path']}`",
                    "",
                ]
            )
    else:
        sections.extend(["Locked test accessed: `false`", ""])
    qualification_paths = sorted(input_dir.rglob("hardware_qualification.json"))
    if qualification_paths:
        qualification = json.loads(qualification_paths[0].read_text(encoding="utf-8"))
        candidates = pd.DataFrame(qualification.get("candidates", []))
        allowed = [
            "model_name",
            "rank_information_coefficient",
            "positive_split_share",
            "validation_splits",
            "beats_matched_classical",
            "ic_advantage_ci_lower",
            "statistically_eligible",
            "hardware_execution_path",
            "qualified_for_hardware",
        ]
        safe_candidates = candidates.loc[
            :, [column for column in allowed if column in candidates]
        ].copy()
        numeric = safe_candidates.select_dtypes(include="number").columns
        safe_candidates[numeric] = safe_candidates[numeric].round(6)
        sections.extend(["## Hardware Qualification", ""])
        sections.append(
            _to_markdown(safe_candidates)
            if not safe_candidates.empty
            else "No QML candidates."
        )
        sections.append("")
    promotion_paths = sorted(input_dir.rglob("qsvm_stability_promotion.json"))
    if promotion_paths:
        promotion = json.loads(promotion_paths[0].read_text(encoding="utf-8"))
        allowed = [
            "candidate",
            "baseline",
            "candidate_rank_information_coefficient",
            "baseline_rank_information_coefficient",
            "positive_split_share",
            "validation_splits",
            "ic_advantage_ci_lower",
            "maximum_positive_gain_contribution",
            "selected_configuration_count",
            "eligible_for_promotion",
            "locked_test_accessed",
        ]
        safe_promotion = pd.DataFrame(
            [{column: promotion.get(column) for column in allowed}]
        )
        numeric = safe_promotion.select_dtypes(include="number").columns
        safe_promotion[numeric] = safe_promotion[numeric].round(6)
        sections.extend(
            ["## QSVM Stability Promotion", "", _to_markdown(safe_promotion), ""]
        )
    for name, allowed in TABLE_COLUMNS.items():
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        columns = [column for column in allowed if column in table]
        safe = table.loc[:, columns].copy()
        numeric = safe.select_dtypes(include="number").columns
        safe[numeric] = safe[numeric].round(6)
        sections.extend([f"## {name.replace('_', ' ').title()}", ""])
        sections.append(_to_markdown(safe) if not safe.empty else "No rows.")
        sections.append("")
    return "\n".join(sections)


def _to_markdown(table: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render_report(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
