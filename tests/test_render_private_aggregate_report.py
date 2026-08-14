import json

import pandas as pd

from scripts.render_private_aggregate_report import render_report


def test_render_report_only_includes_allowlisted_aggregate_columns(tmp_path):
    (tmp_path / "conclusion.json").write_text(
        json.dumps(
            {
                "decision": "Classical remains default.",
                "locked_test_accessed": False,
                "simulator_winner": "qsvm_tuned",
                "hardware_candidate": None,
                "qualified_for_hardware": False,
                "hardware_execution_path": None,
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
    assert "Simulator winner: `qsvm_tuned`" in report
    assert "Qualified for hardware: `False`" in report


def test_render_report_supports_classical_promotion_without_conclusion(tmp_path):
    pd.DataFrame(
        {
            "model_name": ["ranker"],
            "rank_information_coefficient": [0.03],
            "eligible_for_locked_test": [True],
            "decision": ["eligible_to_freeze_for_locked_test"],
            "symbol": ["PRIVATE"],
        }
    ).to_parquet(tmp_path / "research_promotion.parquet", index=False)

    report = render_report(tmp_path)

    assert "eligible_to_freeze_for_locked_test" in report
    assert "Locked test accessed: `false`" in report
    assert "PRIVATE" not in report


def test_render_report_allowlists_nested_hardware_qualification(tmp_path):
    qml = tmp_path / "qml_validation"
    qml.mkdir()
    (qml / "hardware_qualification.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "model_name": "qsvm",
                        "rank_information_coefficient": 0.0341594,
                        "positive_split_share": 2 / 3,
                        "validation_splits": 3,
                        "beats_matched_classical": True,
                        "ic_advantage_ci_lower": -0.01,
                        "statistically_eligible": False,
                        "hardware_execution_path": None,
                        "qualified_for_hardware": False,
                        "private_row_identifier": "DO_NOT_RENDER",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = render_report(tmp_path)

    assert "## Hardware Qualification" in report
    assert "0.034159" in report
    assert "ic_advantage_ci_lower" in report
    assert "DO_NOT_RENDER" not in report
