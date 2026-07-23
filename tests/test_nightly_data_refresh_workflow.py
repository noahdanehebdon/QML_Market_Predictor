from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/nightly-data-refresh.yml")


def _workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_nightly_refresh_supports_manual_and_scheduled_runs():
    workflow = _workflow()

    assert "workflow_dispatch" in workflow["on"]
    assert workflow["on"]["schedule"] == [{"cron": "0 23 * * 1-5"}]


def test_nightly_refresh_uses_secrets_without_echoing_values():
    workflow = _workflow()
    job = workflow["jobs"]["refresh"]
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert job["env"] == {
        "ALPACA_API_KEY": "${{ secrets.ALPACA_API_KEY }}",
        "ALPACA_SECRET_KEY": "${{ secrets.ALPACA_SECRET_KEY }}",
        "BLS_API_KEY": "${{ secrets.BLS_API_KEY }}",
        "SEC_USER_AGENT": "${{ secrets.SEC_USER_AGENT }}",
    }
    assert 'echo "$ALPACA' not in workflow_text
    assert 'echo "$BLS' not in workflow_text
    assert 'echo "$SEC' not in workflow_text


def test_nightly_refresh_runs_all_sources_and_uploads_private_r2_snapshot():
    workflow = _workflow()
    steps = workflow["jobs"]["refresh"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    upload = next(
        step for step in steps if step["name"] == "Upload private R2 snapshot"
    )

    assert "scripts.ingest_alpaca_prices" in commands
    assert "scripts.snapshot_alpaca_assets" in commands
    assert "scripts.build_point_in_time_universe" in commands
    assert "--confirm-provider-permissions" in commands
    assert "scripts.pull_macro" in commands
    assert "--start-year 2020" in commands
    assert "scripts.build_sec_ticker_cik_lookup" in commands
    assert "scripts.ingest_sec_submissions" in commands
    assert "scripts.ingest_sec_company_facts" in commands
    assert "scripts.data_manifest create" in commands
    assert "aws s3 sync data/processed/" in upload["run"]
    assert "processed/runs/${GITHUB_RUN_ID}/" in upload["run"]
    assert "processed/latest-run-id.txt" in upload["run"]
    assert "data/raw/" not in upload["run"]
    assert "actions/upload-artifact" not in WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "git push" not in commands


def test_nightly_refresh_scopes_r2_credentials_to_upload_step():
    workflow = _workflow()
    job = workflow["jobs"]["refresh"]
    upload = next(
        step for step in job["steps"] if step["name"] == "Upload private R2 snapshot"
    )

    assert "AWS_ACCESS_KEY_ID" not in job["env"]
    assert upload["env"]["AWS_ACCESS_KEY_ID"] == "${{ secrets.R2_ACCESS_KEY_ID }}"
    assert (
        upload["env"]["AWS_SECRET_ACCESS_KEY"] == "${{ secrets.R2_SECRET_ACCESS_KEY }}"
    )


def test_nightly_refresh_has_readable_failure_annotation():
    workflow = _workflow()
    steps = workflow["jobs"]["refresh"]["steps"]
    failure_step = next(
        step for step in steps if step["name"] == "Summarize refresh failure"
    )

    assert failure_step["if"] == "failure()"
    assert "::error title=Nightly data refresh failed::" in failure_step["run"]
