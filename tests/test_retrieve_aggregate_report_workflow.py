from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/retrieve-aggregate-report.yml")


def test_retrieval_workflow_only_publishes_rendered_markdown():
    workflow = yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    steps = workflow["jobs"]["report"]["steps"]
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    upload = next(step for step in steps if step.get("name") == "Upload safe report")

    assert workflow["permissions"] == {"contents": "read"}
    assert "definitive_private/" in text
    assert "qml_qualification" in text
    assert "qml_validation/hardware_qualification.json" in text
    assert "qml_validation/qsvm_stability_promotion.json" in text
    assert "run_root" not in text
    assert 'report_prefix=""' not in text
    assert "predictions.parquet" not in text
    assert upload["with"]["path"] == "aggregate-report.md"
    assert "private-aggregate" not in upload["with"]["path"]
