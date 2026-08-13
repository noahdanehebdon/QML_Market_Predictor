"""Evaluate a locally trained VQC on a bounded IBM Quantum validation subset."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.qml.encoding import AngleEncodingConfig, angle_encode_dataset
from market_qml.qml.ibm_backend import (
    IBMBackendConfig,
    collect_ibm_vqc,
    save_ibm_execution,
    save_ibm_submission,
    submit_ibm_vqc,
)
from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.vqc import VariationalQuantumClassifier, _circuit_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--deterministic-fixture", action="store_true")
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help="Simulator qualification JSON required for trained-model hardware inference.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--max-total-shots", type=int, required=True)
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")
    if args.deterministic_fixture:
        rng = np.random.default_rng(122)
        angles = rng.uniform(-np.pi, np.pi, size=(args.rows, 8))
        weights = rng.uniform(-0.1, 0.1, size=(1, 8))
    else:
        if args.model is None or args.sample is None:
            parser.error(
                "--model and --sample are required without --deterministic-fixture"
            )
        if args.qualification_report is None:
            parser.error(
                "--qualification-report is required for trained-model hardware inference"
            )
        qualification = json.loads(
            args.qualification_report.read_text(encoding="utf-8")
        )
        if qualification.get("qualified_for_hardware") is not True:
            raise RuntimeError(
                "Simulator candidate is not qualified for IBM inference."
            )
        with args.model.open("rb") as handle:
            model = pickle.load(handle)
        if (
            not isinstance(model, VariationalQuantumClassifier)
            or model.weights_ is None
        ):
            raise TypeError(
                "--model must contain a fitted VariationalQuantumClassifier"
            )
        data = build_qml_train_validation(
            pd.read_parquet(args.sample), split_id=args.split_id
        )
        validation = data.validation
        angles = (
            angle_encode_dataset(
                validation,
                config=AngleEncodingConfig(n_qubits=model.n_qubits),
                feature_columns=list(validation.X.columns),
            )
            .X.head(args.rows)
            .to_numpy(dtype=float)
        )
        weights = model.weights_
    config = IBMBackendConfig(
        backend_name=args.backend,
        shots=args.shots,
        max_circuits=args.rows,
        max_total_shots=args.max_total_shots,
    )
    submitted = submit_ibm_vqc(angles, weights, config)
    submission_path = save_ibm_submission(submitted, args.output_dir)
    print(f"Submitted IBM Runtime job: {submitted.job_id}")
    print(f"submission: {submission_path}")
    result = collect_ibm_vqc(submitted.job_id, config, metadata=submitted.metadata)
    paths = save_ibm_execution(result, args.output_dir)
    exact_scores = _circuit_probabilities(angles, weights).tolist()
    comparison = [
        {
            "row": index,
            "exact_score": exact,
            "hardware_score": hardware,
            "absolute_difference": abs(exact - hardware),
        }
        for index, (exact, hardware) in enumerate(zip(exact_scores, result.scores))
    ]
    comparison_path = args.output_dir / "simulator_hardware_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    rank_preservation = float(
        pd.Series(exact_scores).corr(pd.Series(result.scores), method="spearman")
    )
    qualification_path = args.output_dir / "hardware_rank_preservation.json"
    qualification_path.write_text(
        json.dumps(
            {
                "spearman_rank_preservation": rank_preservation,
                "rows": len(exact_scores),
                "qualified": bool(rank_preservation >= 0.5),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"IBM Runtime job: {result.job_id}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"comparison: {comparison_path}")
    print(f"rank preservation: {qualification_path}")


if __name__ == "__main__":
    main()
