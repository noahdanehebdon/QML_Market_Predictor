"""Build staged shadow and Alpaca paper validation evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from market_qml.execution.validation import (
    ValidationCriteria,
    build_validation_report,
    save_validation_report,
)

DEFAULT_CONFIG_PATH = Path("configs/validation.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare shadow, paper, valuation, and backtest evidence."
    )
    parser.add_argument("--shadow-record", type=Path, action="append", default=[])
    parser.add_argument(
        "--reconciliation-report", type=Path, action="append", default=[]
    )
    parser.add_argument("--valuations", type=Path, required=True)
    parser.add_argument("--backtest-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_criteria(path: str | Path) -> ValidationCriteria:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("validation"), dict):
        raise ValueError("Validation config must contain a validation mapping.")
    allowed = set(ValidationCriteria.__dataclass_fields__)
    unknown = set(config["validation"]) - allowed
    if unknown:
        raise ValueError("Unknown validation fields: " + ", ".join(sorted(unknown)))
    return ValidationCriteria(**config["validation"])


def main() -> None:
    args = parse_args()
    report = build_validation_report(
        [_load_json(path) for path in args.shadow_record],
        [_load_json(path) for path in args.reconciliation_report],
        pd.read_csv(args.valuations),
        backtest_summary=_load_json(args.backtest_summary),
        generated_at=datetime.now(timezone.utc),
        criteria=load_criteria(args.config),
    )
    save_validation_report(report, args.output)
    print(f"Saved immutable validation report to {args.output}")
    for stage, gate in report["promotion_gates"].items():
        print(f"{stage}: {'eligible' if gate['eligible'] else 'not eligible'}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


if __name__ == "__main__":
    main()
