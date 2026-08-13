from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/replay-portfolio-report.yml")


def test_replay_workflow_reuses_predictions_but_only_publishes_markdown():
    workflow = yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    steps = workflow["jobs"]["replay"]["steps"]
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    upload = next(step for step in steps if step.get("name") == "Upload safe report")

    assert workflow["permissions"] == {"contents": "read"}
    assert "classical_full/predictions.parquet" in text
    assert "qml_tuned_full/predictions.parquet" in text
    assert '--target-horizon-days "${{ inputs.target_horizon_days }}"' in text
    assert upload["with"]["path"] == "replay-report.md"
    assert "predictions.parquet" not in upload["with"]["path"]
