from pathlib import Path

import yaml


WORKFLOW_PATH = Path(".github/workflows/weekly-retraining.yml")


def _workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_weekly_retraining_supports_manual_and_scheduled_runs():
    triggers = _workflow()["on"]

    assert "workflow_dispatch" in triggers
    assert triggers["schedule"] == [{"cron": "0 8 * * 6"}]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["baseline_model"]["default"] == "tuned_gradient_boosting_regressor"
    assert inputs["run_qml"]["default"] == "false"


def test_weekly_retraining_downloads_and_verifies_latest_r2_snapshot():
    workflow = _workflow()
    job = workflow["jobs"]["retrain"]
    steps = job["steps"]
    download = next(step for step in steps if step["name"] == "Download latest private R2 snapshot")

    permissions = job.get("permissions", workflow["permissions"])
    assert permissions["contents"] == "read"
    assert "actions" not in permissions
    assert "processed/latest-run-id.txt" in download["run"]
    assert "processed/runs/${run_id}/" in download["run"]
    assert "data/processed/" in download["run"]
    assert "scripts.data_manifest verify" in download["run"]
    assert "gh run download" not in download["run"]


def test_weekly_retraining_builds_features_and_gates_qml():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)

    assert "build-features" in commands
    assert 'models=("$BASELINE_MODEL")' in commands
    assert 'if [[ "$RUN_QML" == "true" ]]' in commands
    assert "models+=(vqc)" in commands
    assert "--disable-mlflow" in commands


def test_weekly_retraining_uploads_metrics_only():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    upload = next(step for step in steps if step["name"] == "Upload metrics")
    paths = upload["with"]["path"]

    assert upload["uses"] == "actions/upload-artifact@v4"
    assert "classification_metrics.parquet" in paths
    assert "ranking_metrics.parquet" in paths
    assert "portfolio_risk_metrics.parquet" in paths
    assert "predictions.parquet" not in paths
    assert "data/" not in paths
    assert upload["with"]["retention-days"] == "30"


def test_weekly_retraining_has_readable_failure_annotation():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    failure = next(step for step in steps if step["name"] == "Summarize retraining failure")

    assert failure["if"] == "failure()"
    assert "::error title=Weekly retraining failed::" in failure["run"]
