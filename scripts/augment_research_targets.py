"""Add integrity flags and stock-specific residual targets to canonical labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.labels.integrity import ReturnIntegrityRules, add_return_integrity_flags
from market_qml.labels.residualized import build_residualized_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, default=Path("data/labels/forward_return_labels.parquet")
    )
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/feature_table.parquet")
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=Path("data/processed/universe_membership.parquet"),
    )
    parser.add_argument("--target-horizon-days", type=int, required=True)
    parser.add_argument("--maximum-absolute-return", type=float, default=2.0)
    args = parser.parse_args()
    horizon = args.target_horizon_days
    if horizon <= 0:
        raise ValueError("target_horizon_days must be positive.")
    labels = pd.read_parquet(args.labels)
    target = f"forward_excess_return_{horizon}d"
    labels = add_return_integrity_flags(
        labels,
        return_column=f"forward_return_{horizon}d",
        rules=ReturnIntegrityRules(
            maximum_absolute_return=args.maximum_absolute_return
        ),
    )
    exposures = pd.read_parquet(args.features)
    if args.membership.exists():
        membership = pd.read_parquet(args.membership)
        columns = ["symbol", "date"] + [
            column for column in ["sector", "size_bucket"] if column in membership
        ]
        exposures = exposures.merge(
            membership.loc[membership["is_member"].eq(True), columns],
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )
    residual = build_residualized_target(
        labels,
        exposures,
        target_column=target,
    )
    labels = labels.merge(residual, on=["symbol", "date"], validate="one_to_one")
    labels.to_parquet(args.labels, index=False)
    print(
        {
            "rows": len(labels),
            "integrity_valid_share": float(labels["return_integrity_valid"].mean()),
            "residual_target": f"residualized_{target}",
        }
    )


if __name__ == "__main__":
    main()
