"""Build private point-in-time universe membership and diagnostic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from market_qml.universe import (
    UniverseRules,
    build_point_in_time_universe,
    universe_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", type=Path, default=Path("data/processed/prices.parquet")
    )
    parser.add_argument(
        "--asset-history",
        type=Path,
        default=Path("data/processed/asset_history.parquet"),
    )
    parser.add_argument("--metadata-history", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/universe.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/universe_membership.parquet"),
    )
    parser.add_argument(
        "--coverage", type=Path, default=Path("reports/universe_coverage.parquet")
    )
    parser.add_argument(
        "--transitions", type=Path, default=Path("reports/universe_transitions.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/universe_manifest.json")
    )
    parser.add_argument(
        "--confirm-provider-permissions",
        action="store_true",
        help="Acknowledge private research use under the user's current provider terms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_provider_permissions:
        raise SystemExit(
            "Provider permissions must be confirmed for the current account and data plan. "
            "Re-run with --confirm-provider-permissions; this never permits redistribution."
        )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config["point_in_time"]
    rules = UniverseRules(**settings["rules"])
    prices = pd.read_parquet(args.prices)
    assets = pd.read_parquet(args.asset_history)
    metadata = (
        pd.read_parquet(args.metadata_history)
        if args.metadata_history is not None
        else None
    )
    membership = build_point_in_time_universe(
        prices,
        assets,
        metadata_history=metadata,
        rules=rules,
        benchmark_symbol=config["universe"]["benchmark"],
        legacy_seed_symbols=config["universe"]["symbols"],
    )
    coverage, transitions, summary = universe_diagnostics(membership, rules=rules)
    for path in [args.output, args.coverage, args.transitions, args.manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(args.output, index=False)
    coverage.to_parquet(args.coverage, index=False)
    transitions.to_parquet(args.transitions, index=False)
    manifest = {
        **summary,
        "rules": settings["rules"],
        "provider": settings["provider"],
        "provider_permissions_confirmed_by_user": True,
        "redistribution_permitted": False,
        "storage": "private_local_and_r2_only",
        "historical_security_master": settings["limitations"][
            "historical_security_master"
        ],
        "survivorship_limitation": settings["limitations"]["survivorship"],
        "delisting_limitation": settings["limitations"]["delistings"],
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
