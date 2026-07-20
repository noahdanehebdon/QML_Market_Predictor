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
    assert "echo \"$ALPACA" not in workflow_text
    assert "echo \"$BLS" not in workflow_text
    assert "echo \"$SEC" not in workflow_text


def test_nightly_refresh_runs_all_sources_and_uploads_ignored_data():
    workflow = _workflow()
    steps = workflow["jobs"]["refresh"]["steps"]
    commands = "\n".join(step.get("run", "") for step in steps)
    upload = next(step for step in steps if step["name"] == "Upload refreshed data")

    assert "scripts.ingest_alpaca_prices" in commands
    assert "scripts.pull_macro" in commands
    assert "--start-year 2020" in commands
    assert "scripts.build_sec_ticker_cik_lookup" in commands
    assert "scripts.ingest_sec_submissions" in commands
    assert "scripts.ingest_sec_company_facts" in commands
    assert upload["uses"] == "actions/upload-artifact@v4"
    assert "data/processed/" in upload["with"]["path"]
    assert "data/raw/" not in upload["with"]["path"]
    assert "git push" not in commands


def test_nightly_refresh_has_readable_failure_annotation():
    workflow = _workflow()
    steps = workflow["jobs"]["refresh"]["steps"]
    failure_step = next(step for step in steps if step["name"] == "Summarize refresh failure")

    assert failure_step["if"] == "failure()"
    assert "::error title=Nightly data refresh failed::" in failure_step["run"]
