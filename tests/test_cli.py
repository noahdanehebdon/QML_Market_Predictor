from pathlib import Path

import pytest
import yaml

from market_qml.cli import COMMAND_NAMES, load_command_steps, run_command


def _write_config(path: Path, commands: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump({"commands": commands}), encoding="utf-8")
    return path


def test_repository_cli_config_defines_every_command() -> None:
    for command_name in COMMAND_NAMES:
        steps = load_command_steps(Path("configs/cli.yaml"), command_name)
        assert steps
        assert all(step[1] == "-m" for step in steps)


def test_run_command_dry_run_does_not_execute(tmp_path, monkeypatch, capsys) -> None:
    config_path = _write_config(
        tmp_path / "cli.yaml",
        {"train": {"steps": [{"module": "scripts.example", "args": ["--x", 2]}]}},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called during a dry run")

    monkeypatch.setattr("market_qml.cli.subprocess.run", fail_if_called)

    assert run_command("train", ["--config", str(config_path), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "scripts.example" in output
    assert "--x 2" in output


def test_run_command_executes_all_configured_steps(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path / "cli.yaml",
        {
            "report": {
                "steps": [
                    {"module": "scripts.first"},
                    {"module": "scripts.second", "args": ["--output", "report.md"]},
                ]
            }
        },
    )
    executed = []
    monkeypatch.setattr(
        "market_qml.cli.subprocess.run",
        lambda command, check: executed.append((command, check)),
    )

    assert run_command("report", ["--config", str(config_path)]) == 0
    assert [command[0][2] for command in executed] == [
        "scripts.first",
        "scripts.second",
    ]
    assert all(check is True for _, check in executed)


@pytest.mark.parametrize(
    "commands, message",
    [
        ({}, "no steps"),
        ({"backtest": {"steps": []}}, "at least one step"),
        (
            {"backtest": {"steps": [{"module": "scripts.bad;command"}]}},
            "invalid module name",
        ),
    ],
)
def test_invalid_cli_config_is_rejected(tmp_path, commands, message) -> None:
    config_path = _write_config(tmp_path / "cli.yaml", commands)
    with pytest.raises(ValueError, match=message):
        load_command_steps(config_path, "backtest")
