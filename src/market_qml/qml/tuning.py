"""Reproducible architecture and optimizer tuning for the VQC baseline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.qml.interface import QMLModelConfig, QMLTrainValidation
from market_qml.qml.vqc import (
    VariationalQuantumClassifier,
    _binary_cross_entropy,
)


DEFAULT_OVERFIT_GAP_THRESHOLD = 0.05


@dataclass(frozen=True)
class VQCTuningResult:
    """VQC grid-search results, loss histories, and selected configuration."""

    results: pd.DataFrame
    loss_history: pd.DataFrame
    best_config: dict[str, object]


def tune_vqc(
    data: QMLTrainValidation,
    *,
    ansatz_depths: list[int],
    learning_rates: list[float],
    optimizers: list[str],
    max_iter: int = 30,
    n_qubits: int = 8,
    batch_size: int = 32,
    perturbation: float = 0.1,
    l2: float = 0.001,
    random_state: int = 42,
    overfit_gap_threshold: float = DEFAULT_OVERFIT_GAP_THRESHOLD,
) -> VQCTuningResult:
    """Evaluate a deterministic grid and select the lowest validation loss."""
    _validate_grid(ansatz_depths, learning_rates, optimizers)
    if overfit_gap_threshold < 0:
        raise ValueError("overfit_gap_threshold must be non-negative.")

    rows: list[dict[str, object]] = []
    loss_frames: list[pd.DataFrame] = []
    configurations = product(ansatz_depths, learning_rates, optimizers)
    for config_index, (depth, learning_rate, optimizer) in enumerate(configurations):
        config_id = f"vqc_config_{config_index:03d}"
        model = VariationalQuantumClassifier(
            QMLModelConfig(
                model_name=config_id,
                seed=random_state,
                params={
                    "n_qubits": n_qubits,
                    "ansatz_depth": depth,
                    "max_iter": max_iter,
                    "learning_rate": learning_rate,
                    "l2": l2,
                    "perturbation": perturbation,
                    "batch_size": batch_size,
                    "ansatz": "ry_ring_cnot",
                    "simulator": "numpy_statevector",
                    "optimizer": optimizer,
                },
            )
        )
        model.fit(data.train)
        train_targets = data.train.y.to_numpy(dtype=float)
        validation_targets = data.validation.y.to_numpy(dtype=float)
        train_scores = np.asarray(model.predict_scores(data.train), dtype=float)
        validation_scores = np.asarray(
            model.predict_scores(data.validation),
            dtype=float,
        )
        train_loss = _binary_cross_entropy(train_targets, train_scores)
        validation_loss = _binary_cross_entropy(validation_targets, validation_scores)
        validation_predictions = (validation_scores >= 0.5).astype(int)
        overfit_gap = validation_loss - train_loss
        rows.append(
            {
                "config_id": config_id,
                "ansatz_depth": depth,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "max_iter": max_iter,
                "n_qubits": n_qubits,
                "batch_size": batch_size,
                "perturbation": perturbation,
                "l2": l2,
                "random_state": random_state,
                "train_log_loss": train_loss,
                "validation_log_loss": validation_loss,
                "overfit_gap": overfit_gap,
                "overfitting_flag": bool(overfit_gap > overfit_gap_threshold),
                "validation_accuracy": float(
                    (validation_predictions == validation_targets).mean()
                ),
                "validation_brier_score": float(
                    np.mean((validation_scores - validation_targets) ** 2)
                ),
            }
        )
        history = pd.DataFrame(
            {
                "config_id": config_id,
                "iteration": np.arange(1, len(model.loss_history_) + 1),
                "loss": model.loss_history_,
            }
        )
        history["ansatz_depth"] = depth
        history["learning_rate"] = learning_rate
        history["optimizer"] = optimizer
        loss_frames.append(history)

    results = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier_score", "config_id"]
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    best_config = _best_config(results.iloc[0])
    return VQCTuningResult(
        results=results,
        loss_history=pd.concat(loss_frames, ignore_index=True),
        best_config=best_config,
    )


def render_vqc_tuning_report(result: VQCTuningResult) -> str:
    """Render the tuning outcome and selected configuration as Markdown."""
    best = result.best_config
    flagged = int(result.results["overfitting_flag"].sum())
    lines = [
        "# VQC Architecture and Optimizer Tuning",
        "",
        "Configurations are ranked by validation log loss, followed by validation "
        "Brier score. Lower values are better.",
        "",
        "## Best configuration",
        "",
        f"- Configuration: `{best['config_id']}`",
        f"- Ansatz depth: `{best['ansatz_depth']}`",
        f"- Learning rate: `{best['learning_rate']}`",
        f"- Optimizer: `{best['optimizer']}`",
        f"- Training log loss: `{best['train_log_loss']:.6f}`",
        f"- Validation log loss: `{best['validation_log_loss']:.6f}`",
        f"- Overfit gap: `{best['overfit_gap']:.6f}`",
        "",
        "## Overfitting check",
        "",
        f"{flagged} of {len(result.results)} configurations exceeded the configured "
        "validation-minus-training loss threshold.",
        "",
        "## Results",
        "",
        _markdown_table(result.results),
        "",
    ]
    return "\n".join(lines)


def save_vqc_tuning_result(
    result: VQCTuningResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save complete tuning results and the documented best configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "vqc_tuning_results.parquet",
        "loss_history": output_dir / "vqc_tuning_loss_history.parquet",
        "best_config": output_dir / "vqc_best_config.json",
        "report": output_dir / "vqc_tuning_report.md",
    }
    result.results.to_parquet(paths["results"], index=False)
    result.loss_history.to_parquet(paths["loss_history"], index=False)
    paths["best_config"].write_text(
        json.dumps(result.best_config, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(render_vqc_tuning_report(result), encoding="utf-8")
    return paths


def _best_config(row: pd.Series) -> dict[str, object]:
    keys = [
        "config_id",
        "ansatz_depth",
        "learning_rate",
        "optimizer",
        "max_iter",
        "n_qubits",
        "batch_size",
        "perturbation",
        "l2",
        "random_state",
        "train_log_loss",
        "validation_log_loss",
        "overfit_gap",
        "overfitting_flag",
        "validation_accuracy",
        "validation_brier_score",
    ]
    result: dict[str, object] = {}
    for key in keys:
        value = row[key]
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def _validate_grid(
    ansatz_depths: list[int],
    learning_rates: list[float],
    optimizers: list[str],
) -> None:
    if not ansatz_depths or any(depth <= 0 for depth in ansatz_depths):
        raise ValueError("ansatz_depths must contain positive values.")
    if not learning_rates or any(rate <= 0 for rate in learning_rates):
        raise ValueError("learning_rates must contain positive values.")
    if not optimizers:
        raise ValueError("optimizers must not be empty.")


def _markdown_table(data: pd.DataFrame) -> str:
    columns = [
        "rank",
        "config_id",
        "ansatz_depth",
        "learning_rate",
        "optimizer",
        "train_log_loss",
        "validation_log_loss",
        "overfit_gap",
        "overfitting_flag",
        "validation_accuracy",
    ]
    table = data[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
