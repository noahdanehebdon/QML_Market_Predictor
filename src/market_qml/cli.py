"""Configuration-driven command-line entrypoints for project workflows."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/cli.yaml")
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


def load_command_steps(config_path: Path, command_name: str) -> list[list[str]]:
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
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML CLI pipeline configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configured steps without executing them.",
    )
    args = parser.parse_args(argv)

    steps = load_command_steps(args.config, command_name)
    for step in steps:
        print("+", subprocess.list2cmdline(step), flush=True)
        if not args.dry_run:
            subprocess.run(step, check=True)
    return 0


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
