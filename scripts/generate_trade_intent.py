"""Generate a broker-independent paper-trade intent from approved inputs."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from market_qml.execution.trade_intent import (
    PortfolioPolicy,
    build_trade_intent,
    load_promotion_manifest,
    save_trade_intent,
)

DEFAULT_CONFIG_PATH = Path("configs/trade_intent.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an immutable, broker-independent trade intent."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, required=True)
    parser.add_argument("--account-equity", type=float, required=True)
    parser.add_argument(
        "--as-of",
        required=True,
        help="Timezone-aware ISO-8601 decision timestamp, for example 2026-07-23T14:00:00Z.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_policy(path: str | Path) -> PortfolioPolicy:
    """Load the checked-in portfolio policy snapshot."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(
        config.get("portfolio_policy"), dict
    ):
        raise ValueError("Trade-intent config must contain a portfolio_policy mapping.")
    allowed = set(PortfolioPolicy.__dataclass_fields__)
    supplied = set(config["portfolio_policy"])
    unknown = supplied - allowed
    if unknown:
        raise ValueError(
            "Unknown portfolio policy fields: " + ", ".join(sorted(unknown))
        )
    return PortfolioPolicy(**config["portfolio_policy"])


def main() -> None:
    args = parse_args()
    signals = _load_csv(args.signals, "signals")
    positions = _load_csv(args.positions, "positions")
    intent = build_trade_intent(
        signals,
        positions,
        promotion=load_promotion_manifest(args.promotion),
        account_equity=args.account_equity,
        as_of=datetime.fromisoformat(args.as_of.replace("Z", "+00:00")),
        policy=load_policy(args.config),
    )
    save_trade_intent(intent, args.output)
    print(f"Saved immutable trade intent to {args.output}")
    print(f"Run ID: {intent['run_id']}")
    print(f"Proposed trades: {len(intent['trades'])}")
    print("Broker submission allowed: false")


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"{label.capitalize()} file does not exist: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
