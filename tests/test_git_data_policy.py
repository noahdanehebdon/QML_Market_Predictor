from scripts.check_git_data_policy import (
    MAX_FIXTURE_BYTES,
    MAX_TRACKED_BYTES,
    policy_violations,
)


def test_git_data_policy_allows_source_and_small_fixture(tmp_path):
    source = tmp_path / "src" / "module.py"
    fixture = tmp_path / "tests" / "fixtures" / "sample.csv"
    source.parent.mkdir()
    fixture.parent.mkdir(parents=True)
    source.write_text("print('ok')", encoding="utf-8")
    fixture.write_text("value\n1\n", encoding="utf-8")

    assert policy_violations([source, fixture], tmp_path) == []


def test_git_data_policy_rejects_generated_data_and_artifacts(tmp_path):
    processed = tmp_path / "data" / "processed" / "prices.parquet"
    report = tmp_path / "reports" / "metrics.md"
    processed.parent.mkdir(parents=True)
    report.parent.mkdir()
    processed.write_bytes(b"data")
    report.write_text("results", encoding="utf-8")

    violations = policy_violations([processed, report], tmp_path)

    assert any("generated data directory" in item for item in violations)
    assert any("generated data/model file type" in item for item in violations)
    assert any("generated artifact directory" in item for item in violations)


def test_git_data_policy_rejects_oversized_files(tmp_path):
    large = tmp_path / "large.bin"
    fixture = tmp_path / "tests" / "fixtures" / "large.csv"
    fixture.parent.mkdir(parents=True)
    large.write_bytes(b"x" * (MAX_TRACKED_BYTES + 1))
    fixture.write_bytes(b"x" * (MAX_FIXTURE_BYTES + 1))

    violations = policy_violations([large, fixture], tmp_path)

    assert len([item for item in violations if "exceeds" in item]) == 2


def test_repository_currently_satisfies_git_data_policy():
    from pathlib import Path

    from scripts.check_git_data_policy import tracked_paths

    root = Path.cwd()
    assert policy_violations(tracked_paths(root), root) == []
