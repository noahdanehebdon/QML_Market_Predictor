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
    assert inputs["classical_sweep"]["default"] == "false"
    assert inputs["leakage_stress_test"]["default"] == "false"
    assert inputs["phase2_evaluation"]["default"] == "false"
    assert inputs["full_experiment"]["default"] == "false"
    assert inputs["quantum_sample_rows"]["default"] == "512"
    assert inputs["quantum_iterations"]["default"] == "30"
    assert inputs["target_horizon_days"]["default"] == "20"


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
    assert 'if [[ "$CLASSICAL_SWEEP" == "true" ]]' in commands
    assert "residualized_xgboost_ranker" in commands
    assert "random_rank" in commands
    assert "--feature-lag-days 1" in commands
    assert "--permutation-iterations 2000" in commands
    assert "--train-window-days 504" in commands
    assert (
        'if [[ "$RUN_QML" == "true" && "$CLASSICAL_SWEEP" != "true" '
        '&& "$LEAKAGE_STRESS_TEST" != "true" ]]' in commands
    )
    assert "models+=(vqc)" in commands
    assert "--disable-mlflow" in commands


def test_weekly_retraining_audits_features_before_training():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    names = [step["name"] for step in steps]
    commands = "\n".join(step.get("run", "") for step in steps)

    assert names.index("Audit feature quality and predictive stability") < names.index(
        "Retrain selected models"
    )
    assert "scripts.audit_feature_quality" in commands
    assert "--max-rows-per-date 64" in commands
    assert "scripts.augment_research_targets" in commands
    assert (
        "--universe-membership data/processed/universe_membership.parquet" in commands
    )
    assert '--target-horizon-days "$TARGET_HORIZON_DAYS"' in commands
    assert "reports/weekly_retraining/feature_audit" in commands
    audit = next(
        step
        for step in steps
        if step["name"] == "Audit feature quality and predictive stability"
    )
    assert audit["if"] == (
        "env.CLASSICAL_SWEEP != 'true' && env.LEAKAGE_STRESS_TEST != 'true' && "
        "env.PHASE2_EVALUATION != 'true'"
    )


def test_frozen_phase2_contract_is_development_only():
    config = yaml.safe_load(
        Path("configs/phase2_candidate.yaml").read_text(encoding="utf-8")
    )
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert config["registered_before_results"]
    assert not config["locked_test_accessed"]
    assert config["target"]["horizon_days"] == 20
    assert config["validation"]["feature_lag_days"] == 1
    assert config["validation"]["permutation_iterations"] == 2000
    assert config["promotion"]["minimum_rank_ic"] == 0.03
    for model in config["models"]:
        assert model in workflow
    assert "PHASE2_EVALUATION" in workflow


def test_weekly_retraining_materializes_selected_horizon_before_audit():
    steps = _workflow()["jobs"]["retrain"]["steps"]
    names = [step["name"] for step in steps]
    align = next(
        step
        for step in steps
        if step["name"] == "Align labels and splits to selected horizon"
    )

    assert names.index("Align labels and splits to selected horizon") < names.index(
        "Audit feature quality and predictive stability"
    )
    assert "scripts.build_forward_return_labels" in align["run"]
    assert '--horizon "$TARGET_HORIZON_DAYS"' in align["run"]
    assert "scripts.build_walk_forward_splits" in align["run"]
    assert '--target-horizon-days "$TARGET_HORIZON_DAYS"' in align["run"]
    assert '--purge-days "$TARGET_HORIZON_DAYS"' in align["run"]
    assert '--embargo-days "$TARGET_HORIZON_DAYS"' in align["run"]


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


def test_weekly_retraining_full_experiment_is_explicit_and_private():
    workflow = _workflow()
    job = workflow["jobs"]["retrain"]
    steps = job["steps"]
    experiment = next(
        step
        for step in steps
        if step["name"] == "Run full classical and tuned quantum experiment"
    )

    assert job["timeout-minutes"] == "360"
    assert experiment["if"] == "env.FULL_EXPERIMENT == 'true'"
    assert "tuned_gradient_boosting_regressor" in experiment["run"]
    assert "scripts/compare_qml_models.py" in experiment["run"]
    assert "scripts.build_definitive_qml_comparison" in experiment["run"]
    assert (
        "definitive_private"
        not in WORKFLOW_PATH.read_text(encoding="utf-8").split(
            "actions/upload-artifact"
        )[-1]
    )
    assert '--train-rows "$QUANTUM_SAMPLE_ROWS"' in experiment["run"]
    assert '--iterations "$QUANTUM_ITERATIONS"' in experiment["run"]
    assert "actions/upload-artifact" not in WORKFLOW_PATH.read_text(encoding="utf-8")
