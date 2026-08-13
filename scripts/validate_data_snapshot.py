# mypy: disable-error-code=import-untyped
"""Validate and gate a processed private-data snapshot before publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_qml.ingestion.data_quality import validate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/data_quality"))
    args = parser.parse_args()
    result = validate_snapshot(
        prices=pd.read_parquet(args.data_dir / "prices.parquet"),
        assets=pd.read_parquet(args.data_dir / "asset_history.parquet"),
        submissions=pd.read_parquet(args.data_dir / "sec_submissions.parquet"),
        fundamentals=pd.read_parquet(args.data_dir / "fundamentals.parquet"),
        macro_raw=pd.read_parquet(args.raw_dir / "macro_sources.parquet"),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.checks.to_parquet(args.output_dir / "checks.parquet", index=False)
    result.quarantine.to_parquet(args.output_dir / "quarantine.parquet", index=False)
    summary = {
        "schema_version": 1,
        "status": "passed" if result.passed else "failed",
        "checks": len(result.checks),
        "critical_failures": int(
            ((result.checks.severity == "critical") & ~result.checks.passed).sum()
        ),
        "quarantined_rows": len(result.quarantine),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if not result.passed:
        failures = result.checks.loc[
            (result.checks.severity == "critical") & ~result.checks.passed, "check"
        ].tolist()
        raise SystemExit("Critical data-quality failures: " + ", ".join(failures))


if __name__ == "__main__":
    main()
