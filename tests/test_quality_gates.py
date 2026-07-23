from pathlib import Path

WORKFLOW = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")
PROJECT = Path("pyproject.toml").read_text(encoding="utf-8")


def test_ci_enforces_coverage_typing_security_and_distribution_gates() -> None:
    assert "--cov-fail-under=82" in WORKFLOW
    assert "python -m mypy" in WORKFLOW
    assert "python -m pip_audit . --strict" in WORKFLOW
    assert "python -m twine check dist/*" in WORKFLOW
    assert "python -m scripts.verify_wheel_install" in WORKFLOW
    assert "GITHUB_STEP_SUMMARY" in WORKFLOW


def test_quality_tooling_is_development_only_and_threshold_is_documented() -> None:
    runtime, development = PROJECT.split("[project.optional-dependencies]", maxsplit=1)
    assert "pytest-cov" not in runtime
    assert "pip-audit" not in runtime
    assert "mypy" not in runtime
    assert "fail_under = 82" in development
