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
}


def render_report(input_dir: Path) -> str:
    """Render allowlisted aggregate columns without row-level predictions."""
    conclusion = json.loads((input_dir / "conclusion.json").read_text(encoding="utf-8"))
    sections = [
        "# Aggregate classical-versus-quantum results",
        "",
        f"**Decision:** {conclusion['decision']}",
        "",
        f"Locked test accessed: `{conclusion['locked_test_accessed']}`",
        "",
    ]
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
