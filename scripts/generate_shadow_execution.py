"""Archive proposed orders in a no-network shadow mode."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from market_qml.execution.validation import create_shadow_record, save_shadow_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an immutable shadow record without broker access."
    )
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intent = json.loads(args.intent.read_text(encoding="utf-8"))
    record = create_shadow_record(intent, observed_at=datetime.now(timezone.utc))
    save_shadow_record(record, args.output)
    print(f"Saved immutable no-network shadow record to {args.output}")
    print("Submission capability: none")


if __name__ == "__main__":
    main()
