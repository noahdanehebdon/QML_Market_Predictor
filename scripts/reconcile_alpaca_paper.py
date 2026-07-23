"""Reconcile a paper intent against durable journal and Alpaca state."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from market_qml.execution.broker import AlpacaPaperBroker
from market_qml.execution.journal import ExecutionJournal
from market_qml.execution.reconciliation import (
    reconcile_paper_execution,
    save_reconciliation_report,
)
from scripts.execute_alpaca_paper import (
    DEFAULT_CONFIG_PATH,
    ENABLE_ENV,
    KILL_SWITCH_ENV,
    load_execution_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile Alpaca paper orders, fills, and positions."
    )
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument(
        "--cancel-stale-paper",
        action="store_true",
        help="Request cancellation of stale project orders after safety gates pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_execution_policy(args.config)
    intent = json.loads(args.intent.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)
    submission_enabled = os.getenv(ENABLE_ENV, "").lower() == "true"
    kill_switch_active = os.getenv(KILL_SWITCH_ENV, "active").lower() != "inactive"
    broker = AlpacaPaperBroker.from_environment()
    with ExecutionJournal(args.journal) as journal:
        report = reconcile_paper_execution(
            broker,
            journal,
            intent,
            now=now,
            strategy_id=policy.strategy_id,
            cancel_stale=args.cancel_stale_paper,
            cancel_after_minutes=policy.cancel_after_minutes,
            submission_enabled=submission_enabled,
            kill_switch_active=kill_switch_active,
        )
    save_reconciliation_report(
        report,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
    )
    print(f"Saved reconciliation JSON to {args.json_output}")
    print(f"Saved reconciliation Markdown to {args.markdown_output}")
    print(f"Status: {report['status']}")


if __name__ == "__main__":
    main()
