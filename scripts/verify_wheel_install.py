"""Install a wheel and verify every console command outside the source checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

from market_qml.cli import COMMAND_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    return parser.parse_args()


def verify_wheel(wheel: Path) -> None:
    wheel = wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(f"Wheel does not exist: {wheel}")
    with tempfile.TemporaryDirectory(prefix="market-qml-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        workspace = root / "workspace"
        workspace.mkdir()
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts_dir = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts_dir / ("python.exe" if os.name == "nt" else "python")
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheel)],
            check=True,
            cwd=workspace,
        )
        for command_name in COMMAND_NAMES:
            executable = scripts_dir / (
                f"{command_name}.exe" if os.name == "nt" else command_name
            )
            subprocess.run([str(executable), "--help"], check=True, cwd=workspace)
            subprocess.run(
                [
                    str(executable),
                    "--workspace-root",
                    str(workspace),
                    "--dry-run",
                ],
                check=True,
                cwd=workspace,
            )


def main() -> None:
    verify_wheel(parse_args().wheel)
    print("Wheel install verification passed for all seven console commands.")


if __name__ == "__main__":
    main()
