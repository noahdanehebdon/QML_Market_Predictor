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
    download = next(
        step for step in steps if step["name"] == "Download latest private R2 snapshot"
    )

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


def test_weekly_retraining_uploads_reports_only_to_private_r2():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    upload = next(
        step for step in steps if step["name"] == "Upload reports to private R2"
    )
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "actions/upload-artifact" not in workflow_text
    assert "reports/weekly_retraining/" in upload["run"]
    assert "reports/runs/${GITHUB_RUN_ID}/" in upload["run"]
    assert upload["env"]["AWS_ACCESS_KEY_ID"] == "${{ secrets.R2_ACCESS_KEY_ID }}"
    assert (
        upload["env"]["AWS_SECRET_ACCESS_KEY"] == "${{ secrets.R2_SECRET_ACCESS_KEY }}"
    )
    assert "data/" not in upload["run"]

    retrain = next(step for step in steps if step["name"] == "Retrain selected models")
    assert "--output-dir reports/weekly_retraining" in retrain["run"]


def test_weekly_retraining_has_readable_failure_annotation():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    failure = next(
        step for step in steps if step["name"] == "Summarize retraining failure"
    )

    assert failure["if"] == "failure()"
    assert "::error title=Weekly retraining failed::" in failure["run"]
