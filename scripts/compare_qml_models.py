"""Run issue #49's controlled QML comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from market_qml.qml.comparison import (
    ComparisonConfig,
    run_model_comparison,
    save_comparison_result,
)
from market_qml.qml.selected_features import build_selected_qml_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare QML models with logistic regression and gradient boosting on aligned inputs."
    )
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
    parser.add_argument(
        "--classical-selection",
        type=Path,
        default=Path("reports/backtests/selection_diagnostics.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/qml_comparison")
    )
    parser.add_argument("--train-rows", type=int, default=256)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    args = parser.parse_args()
    features, labels = pd.read_parquet(args.features), pd.read_parquet(args.labels)
    splits = pd.read_parquet(args.splits).sort_values("split_id")
    if args.max_splits is not None:
        splits = splits.head(args.max_splits)
    selected = build_selected_qml_features(
        features=features,
        labels=labels,
        splits=splits,
        selection_diagnostics=pd.read_parquet(args.classical_selection),
    )
    data = selected.features
    result = run_model_comparison(
        data,
        ComparisonConfig(
            train_rows=args.train_rows,
            validation_rows=args.validation_rows,
            vqc_iterations=args.iterations,
            qcnn_iterations=args.iterations,
            transaction_cost_bps=args.transaction_cost_bps,
            rebalance_frequency=args.rebalance_frequency,
        ),
    )
    paths = save_comparison_result(result, args.output_dir)
    selected_feature_path = args.output_dir / "selected_feature_manifest.parquet"
    selected_input_path = args.output_dir / "selected_qml_inputs.parquet"
    selected.manifest.to_parquet(selected_feature_path, index=False)
    selected.features.to_parquet(selected_input_path, index=False)
    print(paths["report"])


if __name__ == "__main__":
    main()
