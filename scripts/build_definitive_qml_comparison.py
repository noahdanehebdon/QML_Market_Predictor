"""Build the two-lane definitive classical-versus-quantum report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from market_qml.qml.definitive_comparison import (
    QML_MODELS,
    build_definitive_comparison,
    save_definitive_comparison,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equal-dir", type=Path, required=True)
    parser.add_argument("--best-classical-dir", type=Path, required=True)
    parser.add_argument("--best-qml-dir", type=Path, required=True)
    parser.add_argument("--target-horizon-days", type=int, required=True)
    parser.add_argument("--qualification-report", type=Path, default=None)
    parser.add_argument(
        "--locked-manifest",
        type=Path,
        default=Path("data/processed/locked_test_manifest.json"),
    )
    parser.add_argument(
        "--private-output",
        type=Path,
        default=Path("reports/weekly_retraining/definitive_private"),
    )
    parser.add_argument(
        "--public-output",
        type=Path,
        default=Path("reports/weekly_retraining/definitive_aggregate"),
    )
    args = parser.parse_args()
    equal = pd.read_parquet(args.equal_dir / "predictions.parquet")
    classical = _only_classical_predictions(
        pd.read_parquet(args.best_classical_dir / "predictions.parquet")
    )
    qml = pd.read_parquet(args.best_qml_dir / "predictions.parquet")
    qml = qml.loc[qml["model_name"].isin(QML_MODELS)]
    best = pd.concat([classical, qml], ignore_index=True)
    manifest = (
        json.loads(args.locked_manifest.read_text())
        if args.locked_manifest.exists()
        else {"locked_test_accessed": False}
    )
    equal_resources_path = args.equal_dir / "resource_usage.parquet"
    best_resources_path = args.best_qml_dir / "resource_usage.parquet"
    result = build_definitive_comparison(
        equal,
        best,
        equal_resources=pd.read_parquet(equal_resources_path)
        if equal_resources_path.exists()
        else None,
        best_resources=pd.read_parquet(best_resources_path)
        if best_resources_path.exists()
        else None,
        locked_test_manifest=manifest,
        return_horizon_days=args.target_horizon_days,
        rebalance_frequency=args.target_horizon_days,
    )
    if args.qualification_report and args.qualification_report.exists():
        qualification = json.loads(args.qualification_report.read_text())
        result.conclusion.update(
            {
                "simulator_winner": qualification.get("simulator_winner"),
                "hardware_candidate": qualification.get("candidate"),
                "qualified_for_hardware": bool(
                    qualification.get("qualified_for_hardware", False)
                ),
                "hardware_execution_path": qualification.get("hardware_execution_path"),
            }
        )
    save_definitive_comparison(result, args.private_output, args.public_output)
    print(result.conclusion["decision"])


def _only_classical_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Prevent QML rows from leaking through a directory used as a control lane."""
    return predictions.loc[~predictions["model_name"].isin(QML_MODELS)].copy()


if __name__ == "__main__":
    main()
