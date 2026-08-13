"""Fail closed unless the downloaded snapshot passed its quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/data_quality/summary.json"),
    )
    args = parser.parse_args()
    if not args.report.exists():
        raise SystemExit(f"Missing data-quality report: {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "passed":
        raise SystemExit(
            "Snapshot data-quality report is missing, incompatible, or failed."
        )
    print(f"Verified passing data-quality report: {args.report}")


if __name__ == "__main__":
    main()
