"""Dry-run or explicitly submit a validated intent to Alpaca paper trading."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from market_qml.execution.broker import (
    ALPACA_PAPER_BASE_URL,
    AlpacaPaperBroker,
    BrokerError,
)
from market_qml.execution.paper_execution import (
    PaperExecutionPolicy,
    PreTradeError,
    cancel_stale_paper_orders,
    execute_paper_intent,
)

DEFAULT_CONFIG_PATH = Path("configs/paper_execution.yaml")
ENABLE_ENV = "MARKET_QML_ENABLE_PAPER_ORDERS"
KILL_SWITCH_ENV = "MARKET_QML_PAPER_KILL_SWITCH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an intent against Alpaca paper state; defaults to dry-run."
    )
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--submit-paper",
        action="store_true",
        help="Submit only after environment enablement and kill-switch checks.",
    )
    parser.add_argument(
        "--cancel-stale-paper",
        action="store_true",
        help="Also cancel stale project-created paper orders after guarded execution.",
    )
    return parser.parse_args()


def load_execution_policy(path: str | Path) -> PaperExecutionPolicy:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(
        config.get("paper_execution"), dict
    ):
        raise ValueError("Execution config must contain a paper_execution mapping.")
    allowed = set(PaperExecutionPolicy.__dataclass_fields__)
    unknown = set(config["paper_execution"]) - allowed
    if unknown:
        raise ValueError(
            "Unknown paper execution fields: " + ", ".join(sorted(unknown))
        )
    return PaperExecutionPolicy(**config["paper_execution"])


def main() -> None:
    args = parse_args()
    policy = load_execution_policy(args.config)
    now = datetime.now(timezone.utc)
    submission_enabled = os.getenv(ENABLE_ENV, "").lower() == "true"
    kill_switch_active = os.getenv(KILL_SWITCH_ENV, "active").lower() != "inactive"
    try:
        intent = json.loads(args.intent.read_text(encoding="utf-8"))
        broker = AlpacaPaperBroker.from_environment(base_url=ALPACA_PAPER_BASE_URL)
        cancellations = []
        if args.cancel_stale_paper:
            if not args.submit_paper:
                raise PreTradeError(
                    "cancel_requires_submission_gate",
                    "Stale cancellation requires --submit-paper.",
                )
            cancellations = cancel_stale_paper_orders(
                broker,
                now=now,
                cancel_after_minutes=policy.cancel_after_minutes,
                submission_enabled=submission_enabled,
                kill_switch_active=kill_switch_active,
            )
        report = execute_paper_intent(
            broker,
            intent,
            now=now,
            policy=policy,
            submit=args.submit_paper,
            submission_enabled=submission_enabled,
            kill_switch_active=kill_switch_active,
        )
        if args.cancel_stale_paper:
            report["cancellations"] = cancellations
    except (
        BrokerError,
        PreTradeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as error:
        report = _blocked_report(error, now)
    _save_private_report(report, args.output)
    print(f"Saved private paper-execution report to {args.output}")
    print(f"Status: {report['status']}")
    print(f"Mode: {report.get('mode', 'blocked')}")
    if report["status"] not in {"approved"}:
        raise SystemExit(1)


def _blocked_report(error: Exception, now: datetime) -> dict[str, Any]:
    code = error.code if isinstance(error, PreTradeError) else "invalid_configuration"
    return {
        "schema_version": 1,
        "mode": "blocked",
        "status": "blocked",
        "paper_only": True,
        "checked_at_utc": now.isoformat(),
        "error": {"code": code, "message": str(error)},
    }


def _save_private_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
