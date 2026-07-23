"""Append today's Alpaca asset state to the private point-in-time history."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from market_qml.ingestion.prices import fetch_alpaca_asset_snapshot

DEFAULT_OUTPUT = Path("data/processed/asset_history.parquet")


def append_asset_snapshot(snapshot: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Append one effective-dated snapshot without rewriting earlier states."""
    if output.exists():
        history = pd.read_parquet(output)
        combined = pd.concat([history, snapshot], ignore_index=True)
    else:
        combined = snapshot.copy()
    combined = combined.drop_duplicates(
        ["symbol", "effective_date"], keep="last"
    ).sort_values(["symbol", "effective_date"])
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshot-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    snapshot = fetch_alpaca_asset_snapshot(snapshot_date=args.snapshot_date)
    history = append_asset_snapshot(snapshot, args.output)
    print(f"Saved {len(snapshot):,} current asset states to private history.")
    print(f"History rows: {len(history):,}; path: {args.output}")


if __name__ == "__main__":
    main()
