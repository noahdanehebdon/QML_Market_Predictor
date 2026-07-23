"""Configuration-driven command-line entrypoints for project workflows."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

WORKSPACE_ENVIRONMENT_VARIABLE = "MARKET_QML_WORKSPACE"
COMMAND_NAMES = (
    "ingest-prices",
    "ingest-macro",
    "ingest-sec",
    "build-features",
    "train",
    "backtest",
    "report",
)
_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def default_config_resource() -> Any:
    """Return the packaged default CLI configuration resource."""
    return files("market_qml").joinpath("default_cli.yaml")


def load_command_steps(config_path: Any, command_name: str) -> list[list[str]]:
    """Load and validate module invocations for one command."""
    if command_name not in COMMAND_NAMES:
        raise ValueError(f"Unknown command: {command_name}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(config.get("commands"), dict):
        raise ValueError("CLI config must contain a 'commands' mapping.")

    command_config = config["commands"].get(command_name)
    if not isinstance(command_config, dict) or not isinstance(
        command_config.get("steps"), list
    ):
        raise ValueError(f"CLI config has no steps for command '{command_name}'.")

    steps: list[list[str]] = []
    for index, step in enumerate(command_config["steps"], start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} for '{command_name}' must be a mapping.")
        module = step.get("module")
        arguments = step.get("args", [])
        if not isinstance(module, str) or not _MODULE_PATTERN.fullmatch(module):
            raise ValueError(
                f"Step {index} for '{command_name}' has an invalid module name."
            )
        if not isinstance(arguments, list) or not all(
            isinstance(argument, (str, int, float)) for argument in arguments
        ):
            raise ValueError(
                f"Step {index} for '{command_name}' must have a scalar args list."
            )
        steps.append([sys.executable, "-m", module, *(str(arg) for arg in arguments)])

    if not steps:
        raise ValueError(f"Command '{command_name}' must contain at least one step.")
    return steps


def run_command(command_name: str, argv: Sequence[str] | None = None) -> int:
    """Parse common options and execute a configured workflow."""
    parser = argparse.ArgumentParser(
        prog=command_name,
        description=f"Run the configured {command_name} workflow.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML pipeline config; defaults to the copy embedded in the package.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=None,
        help=(
            "Root for relative configs, generated data, and artifacts. Defaults to "
            f"${WORKSPACE_ENVIRONMENT_VARIABLE} or the current directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configured steps without executing them.",
    )
    args = parser.parse_args(argv)

    workspace_root = _workspace_root(args.workspace_root)
    config_path = _resolve_config_path(args.config, workspace_root)
    with as_file(config_path) as resolved_config:
        steps = load_command_steps(resolved_config, command_name)
    for step in steps:
        print("+", subprocess.list2cmdline(step), flush=True)
        if not args.dry_run:
            subprocess.run(step, check=True, cwd=workspace_root)
    return 0


def _workspace_root(explicit_root: Path | None) -> Path:
    configured = explicit_root or os.environ.get(WORKSPACE_ENVIRONMENT_VARIABLE)
    root = Path(configured) if configured else Path.cwd()
    return root.expanduser().resolve()


def _resolve_config_path(config_path: Path | None, workspace_root: Path) -> Any:
    if config_path is None:
        return default_config_resource()
    if config_path.is_absolute():
        return config_path
    return workspace_root / config_path


def ingest_prices() -> int:
    return run_command("ingest-prices")


def ingest_macro() -> int:
    return run_command("ingest-macro")


def ingest_sec() -> int:
    return run_command("ingest-sec")


def build_features() -> int:
    return run_command("build-features")


def train() -> int:
    return run_command("train")


def backtest() -> int:
    return run_command("backtest")


def report() -> int:
    return run_command("report")
