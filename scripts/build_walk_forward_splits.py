"""Build walk-forward validation split metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from market_qml.backtest.splits import (
    DEFAULT_SPLIT_OUTPUT_PATH,
    build_walk_forward_split_table,
)
from market_qml.models.dataset import (
    DEFAULT_FEATURE_PATH,
    DEFAULT_LABEL_PATH,
    load_modeling_dataset,
)


DEFAULT_BACKTEST_CONFIG_PATH = Path("configs/backtest.yaml")


def load_walk_forward_config(config_path: Path = DEFAULT_BACKTEST_CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Backtest config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    walk_forward = config.get("walk_forward")
    if not isinstance(walk_forward, dict):
        raise ValueError(f"Missing 'walk_forward' section in {config_path}")

    return walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-ordered walk-forward validation split metadata."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURE_PATH,
        help="Path to canonical feature table parquet.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABEL_PATH,
        help="Path to forward return label table parquet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BACKTEST_CONFIG_PATH,
        help="Path to backtest YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT_PATH,
        help="Path to save walk-forward split metadata parquet.",
    )
    parser.add_argument(
        "--yearly-validation",
        action="store_true",
        help="Use calendar-year validation windows instead of fixed day windows.",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=None,
        help="Trading-day step between fixed validation windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    walk_forward = load_walk_forward_config(args.config)
    dataset = load_modeling_dataset(
        feature_path=args.features,
        label_path=args.labels,
    )

    splits = build_walk_forward_split_table(
        metadata=dataset.metadata,
        output_path=args.output,
        train_window_days=int(walk_forward.get("train_window_days", 756)),
        validation_window_days=int(walk_forward.get("validation_window_days", 126)),
        step_days=args.step_days,
        yearly_validation=args.yearly_validation,
        purge_days=int(walk_forward.get("purge_days", 0)),
    )

    print(f"Saved walk-forward split metadata to {args.output}")
    print(f"Splits: {len(splits)}")
    print("\nColumns:")
    print(list(splits.columns))
    print("\nHead:")
    print(splits.head())
    print("\nTail:")
    print(splits.tail())


if __name__ == "__main__":
    main()
