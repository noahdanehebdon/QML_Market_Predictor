from pathlib import Path

from market_qml.cli import COMMAND_NAMES, load_command_steps

README = Path("README.md")
CLI_CONFIG = Path("configs/cli.yaml")


def test_readme_covers_issue_65_usage_topics():
    text = README.read_text(encoding="utf-8")

    for heading in [
        "## Five-minute overview",
        "## Research snapshot",
        "### Prediction targets",
        "### Data sources",
        "### Model suite",
        "## Local Environment",
        "### Quick start",
        "## Testing",
        "## Limitations",
    ]:
        assert heading in text

    for command in COMMAND_NAMES:
        assert command in text

    assert "python -m scripts.generate_demo_prices" in text
    assert "docs/results_status.md" in text


def test_documented_cli_commands_exist_in_pipeline_config():
    for command in COMMAND_NAMES:
        assert load_command_steps(CLI_CONFIG, command)


def test_readme_does_not_repeat_completed_ci_todo():
    text = README.read_text(encoding="utf-8")

    assert "Add GitHub Actions so tests run automatically" not in text
