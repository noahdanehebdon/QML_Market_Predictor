"""Run issue #49's controlled QML comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    parser.add_argument("--universe-membership", type=Path, default=None)
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
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--validation-rows", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--max-splits", type=int, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--target-horizon-days",
        type=int,
        default=20,
        help="Forward target horizon; selects matching label and return columns.",
    )
    parser.add_argument("--rebalance-frequency", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded worker count for independent QML tuning candidates.",
    )
    args = parser.parse_args()
    features, labels = pd.read_parquet(args.features), pd.read_parquet(args.labels)
    splits = pd.read_parquet(args.splits).sort_values("split_id")
    if args.max_splits is not None:
        splits = splits.head(args.max_splits)
    selected = build_selected_qml_features(
        features=features,
        labels=labels,
        universe_membership=(
            pd.read_parquet(args.universe_membership)
            if args.universe_membership is not None
            else None
        ),
        splits=splits,
        selection_diagnostics=pd.read_parquet(args.classical_selection),
        target_horizon_days=args.target_horizon_days,
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
            rebalance_frequency=(
                args.rebalance_frequency
                if args.rebalance_frequency is not None
                else args.target_horizon_days
            ),
            return_horizon_days=args.target_horizon_days,
            inner_purge_days=args.target_horizon_days,
            max_workers=args.workers,
        ),
    )
    paths = save_comparison_result(result, args.output_dir)
    selected_feature_path = args.output_dir / "selected_feature_manifest.parquet"
    selected_input_path = args.output_dir / "selected_qml_inputs.parquet"
    selected.manifest.to_parquet(selected_feature_path, index=False)
    selected.features.to_parquet(selected_input_path, index=False)
    _write_hardware_qualification(result.ranking_metrics, args.output_dir)
    print(paths["report"])


def _write_hardware_qualification(metrics: pd.DataFrame, output_dir: Path) -> Path:
    split = metrics.loc[metrics["scope"].eq("split")].copy()
    qml_models = {"vqc", "vqc_stable_rank", "qcnn", "qsvm", "qsvm_tuned"}
    hardware_execution_paths = {"vqc": "ibm_vqc"}
    controls = split.loc[split["model_name"].isin(["linear_svm", "rbf_svm"])]
    control_means = controls.groupby("model_name")[
        "rank_information_coefficient"
    ].mean()
    best_control_model = str(control_means.idxmax())
    best_control_ic = float(control_means.max())
    candidates = []
    for model_name, candidate in split.loc[
        split["model_name"].isin(qml_models)
    ].groupby("model_name"):
        candidate_ic = float(candidate["rank_information_coefficient"].mean())
        positive_share = float(candidate["rank_information_coefficient"].gt(0).mean())
        candidate_advantages = _matched_ic_advantages(
            candidate,
            controls.loc[controls["model_name"].eq(best_control_model)],
        )
        advantage_ci_lower = _bootstrap_mean_lower_bound(candidate_advantages)
        statistically_eligible = bool(
            len(candidate) >= 2
            and candidate_ic > 0
            and positive_share >= 2 / 3
            and candidate_ic > best_control_ic
            and advantage_ci_lower > 0
        )
        execution_path = hardware_execution_paths.get(str(model_name))
        candidates.append(
            {
                "model_name": str(model_name),
                "rank_information_coefficient": candidate_ic,
                "positive_split_share": positive_share,
                "validation_splits": len(candidate),
                "beats_matched_classical": bool(candidate_ic > best_control_ic),
                "ic_advantage_ci_lower": advantage_ci_lower,
                "statistically_eligible": statistically_eligible,
                "hardware_execution_path": execution_path,
                "qualified_for_hardware": bool(
                    statistically_eligible and execution_path is not None
                ),
            }
        )
    candidates.sort(key=lambda row: row["rank_information_coefficient"], reverse=True)
    simulator_winner = candidates[0] if candidates else None
    hardware_candidate = next(
        (row for row in candidates if row["qualified_for_hardware"]), None
    )
    report = {
        "candidate": (hardware_candidate["model_name"] if hardware_candidate else None),
        "simulator_winner": (
            simulator_winner["model_name"] if simulator_winner else None
        ),
        "rank_information_coefficient": (
            hardware_candidate["rank_information_coefficient"]
            if hardware_candidate
            else None
        ),
        "positive_split_share": (
            hardware_candidate["positive_split_share"] if hardware_candidate else None
        ),
        "best_matched_classical_ic": best_control_ic,
        "qualified_for_hardware": hardware_candidate is not None,
        "hardware_execution_path": (
            hardware_candidate["hardware_execution_path"]
            if hardware_candidate
            else None
        ),
        "locked_test_accessed": False,
        "candidates": candidates,
        "criteria": {
            "minimum_splits": 2,
            "minimum_positive_split_share": 2 / 3,
            "must_beat_matched_classical": True,
            "minimum_ic_advantage_ci_lower": 0.0,
            "requires_supported_execution_path": True,
        },
    }
    path = output_dir / "hardware_qualification.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _matched_ic_advantages(
    candidate: pd.DataFrame, control: pd.DataFrame
) -> np.ndarray:
    if "split_id" in candidate and "split_id" in control:
        paired = candidate[["split_id", "rank_information_coefficient"]].merge(
            control[["split_id", "rank_information_coefficient"]],
            on="split_id",
            suffixes=("_candidate", "_control"),
            validate="one_to_one",
        )
        return (
            paired["rank_information_coefficient_candidate"]
            - paired["rank_information_coefficient_control"]
        ).to_numpy(float)
    count = min(len(candidate), len(control))
    return (
        candidate["rank_information_coefficient"].to_numpy(float)[:count]
        - control["rank_information_coefficient"].to_numpy(float)[:count]
    )


def _bootstrap_mean_lower_bound(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("-inf")
    rng = np.random.default_rng(42)
    samples = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025))


if __name__ == "__main__":
    main()
