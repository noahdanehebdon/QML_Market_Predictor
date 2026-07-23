"""Deliberately authorize and audit a final locked-test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.backtest.validation import log_locked_test_access


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log deliberate access to the final locked test period."
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/walk_forward_splits.parquet"),
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/validation/locked_test_access.json"),
    )
    args = parser.parse_args()
    splits = pd.read_parquet(args.splits)
    required = {
        "protocol_version",
        "development_end_date",
        "locked_test_start_date",
        "locked_test_end_date",
        "locked_test_days",
        "embargo_days",
        "locked_test_accessed",
    }
    missing = required - set(splits)
    if missing:
        raise ValueError(
            "Split metadata has no locked-test protocol: " + ", ".join(sorted(missing))
        )
    manifest = {column: splits.iloc[0][column] for column in required}
    record = log_locked_test_access(
        manifest, reason=args.reason, audit_path=args.audit_output
    )
    print(f"Locked-test access logged at {args.audit_output}")
    print(
        f"Authorized period: {record['locked_test_start_date']} through "
        f"{record['locked_test_end_date']}"
    )


if __name__ == "__main__":
    main()
