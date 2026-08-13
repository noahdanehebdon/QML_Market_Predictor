import json

import pandas as pd

from scripts.render_private_aggregate_report import render_report


def test_render_report_only_includes_allowlisted_aggregate_columns(tmp_path):
    (tmp_path / "conclusion.json").write_text(
        json.dumps(
            {
                "decision": "Classical remains default.",
                "locked_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "lane": ["best_available"],
            "model_name": ["vqc"],
            "model_family": ["qml"],
            "roc_auc": [0.51234567],
            "symbol": ["PRIVATE"],
            "date": ["2024-01-01"],
        }
    ).to_parquet(tmp_path / "aggregate_metrics.parquet", index=False)

    report = render_report(tmp_path)

    assert "0.512346" in report
    assert "PRIVATE" not in report
    assert "2024-01-01" not in report
    assert "Classical remains default." in report
