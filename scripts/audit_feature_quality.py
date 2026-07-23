"""Produce machine-readable leakage-safe feature audit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from market_qml.features.audit import audit_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Path("data/features/feature_table.parquet")
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/labels/forward_return_labels.parquet")
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/walk_forward_splits.parquet"),
    )
    parser.add_argument("--membership", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/feature_audit")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_features(
        pd.read_parquet(args.features),
        pd.read_parquet(args.labels),
        pd.read_parquet(args.splits),
        membership=pd.read_parquet(args.membership) if args.membership else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: getattr(result, name) for name in result.__dataclass_fields__}
    for name, table in tables.items():
        table.to_parquet(args.output_dir / f"{name}.parquet", index=False)
    manifest = {
        "features": int(result.stability.shape[0]),
        "splits": int(result.quality["split_id"].nunique()),
        "stable_features": int(
            result.stability.get("stable_evidence", pd.Series(dtype=bool)).sum()
        ),
        "artifacts": sorted(f"{name}.parquet" for name in tables),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
