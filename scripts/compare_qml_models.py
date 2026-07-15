"""Run issue #49's controlled QML comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from market_qml.qml.comparison import ComparisonConfig, run_model_comparison, save_comparison_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VQC, QSVM, QCNN, and classical SVM controls.")
    parser.add_argument("--features", type=Path, default=Path("data/features/qml_classification_grouped_pca_features.parquet"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels/forward_return_labels.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/qml_comparison"))
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    features, labels = pd.read_parquet(args.features), pd.read_parquet(args.labels)
    return_columns = ["symbol", "date", "forward_return_5d", "forward_excess_return_5d"]
    data = features.merge(labels[return_columns], on=["symbol", "date"], how="left", validate="many_to_one")
    if data[return_columns[2:]].isna().any().any():
        raise ValueError("Some feature rows did not match forward-return labels.")
    result = run_model_comparison(data, ComparisonConfig(train_rows=args.train_rows,
        validation_rows=args.validation_rows, vqc_iterations=args.iterations, qcnn_iterations=args.iterations))
    paths = save_comparison_result(result, args.output_dir)
    print(paths["report"])


if __name__ == "__main__":
    main()
