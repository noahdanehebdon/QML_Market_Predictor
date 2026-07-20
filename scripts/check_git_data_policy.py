"""Reject generated datasets, model artifacts, and large files tracked by Git."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


MAX_TRACKED_BYTES = 5 * 1024 * 1024
MAX_FIXTURE_BYTES = 1024 * 1024
GENERATED_ROOTS = {"artifacts", "checkpoints", "mlruns", "models", "reports", "runs"}
GENERATED_DATA_DIRS = {
    "data/features",
    "data/labels",
    "data/processed",
    "data/raw",
}
DATA_SUFFIXES = {
    ".csv", ".db", ".feather", ".h5", ".hdf5", ".joblib", ".onnx",
    ".parquet", ".pkl", ".pt", ".pth", ".sqlite", ".sqlite3",
}


def tracked_paths(root: Path) -> list[Path]:
    """Return paths tracked by the repository at root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / path for path in result.stdout.decode().split("\0") if path]


def policy_violations(paths: list[Path], root: Path) -> list[str]:
    """Return policy violations for tracked paths."""
    violations = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name == ".gitkeep":
            continue
        is_fixture = relative.startswith("tests/fixtures/")
        size_limit = MAX_FIXTURE_BYTES if is_fixture else MAX_TRACKED_BYTES
        if path.stat().st_size > size_limit:
            violations.append(f"{relative}: exceeds {size_limit} tracked bytes")
        if is_fixture:
            continue
        if relative.split("/", 1)[0] in GENERATED_ROOTS:
            violations.append(f"{relative}: generated artifact directory")
        if any(relative.startswith(f"{directory}/") for directory in GENERATED_DATA_DIRS):
            violations.append(f"{relative}: generated data directory")
        if path.suffix.lower() in DATA_SUFFIXES:
            violations.append(f"{relative}: generated data/model file type")
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    violations = policy_violations(tracked_paths(root), root)
    if violations:
        raise SystemExit("Git data policy violations:\n- " + "\n- ".join(violations))
    print("Git data policy passed: no generated or oversized files are tracked.")


if __name__ == "__main__":
    main()
