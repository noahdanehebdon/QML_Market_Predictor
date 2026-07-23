"""Build and evaluate multi-horizon prediction targets without opening the test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_qml.labels.forward_returns import build_multi_horizon_target_table
from market_qml.labels.target_research import research_target_candidates, target_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices", type=Path, default=Path("data/processed/prices.parquet")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/labels/target_candidates.parquet")
    )
    parser.add_argument(
        "--diagnostics", type=Path, default=Path("reports/target_diagnostics.parquet")
    )
    parser.add_argument(
        "--selection", type=Path, default=Path("reports/target_selection.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/target_research_manifest.json")
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20, 60])
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--neutral-threshold", type=float, default=0.005)
    parser.add_argument("--sector-column")
    parser.add_argument("--locked-test-days", type=int, default=252)
    parser.add_argument("--embargo-days", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = pd.read_parquet(args.prices)
    labels = build_multi_horizon_target_table(
        prices,
        horizons=args.horizons,
        benchmark_symbol=args.benchmark,
        neutral_threshold=args.neutral_threshold,
        sector_column=args.sector_column,
    )
    diagnostics, selection, manifest = research_target_candidates(
        labels,
        locked_test_days=args.locked_test_days,
        embargo_days=args.embargo_days,
        benchmark=args.benchmark,
    )
    for path in [args.labels, args.diagnostics, args.selection, args.manifest]:
        path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.labels, index=False)
    diagnostics.to_parquet(args.diagnostics, index=False)
    selection.to_parquet(args.selection, index=False)
    args.manifest.write_text(
        json.dumps(
            {
                **manifest,
                "targets": target_catalog(labels, benchmark=args.benchmark).to_dict(
                    "records"
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(selection.to_string(index=False))
    print(f"Locked-test rows inspected: {manifest['locked_test_rows_inspected']}")


if __name__ == "__main__":
    main()
