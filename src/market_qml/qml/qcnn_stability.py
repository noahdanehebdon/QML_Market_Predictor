"""Reproducible QCNN stability experiments and failure-mode diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.qml.interface import QMLDataset, QMLTrainValidation
from market_qml.qml.qcnn import QCNNResult, train_qcnn

DEFAULT_GRADIENT_FLOOR = 1e-5
DEFAULT_GRADIENT_CEILING = 5.0
DEFAULT_LOSS_VOLATILITY_CEILING = 0.2
DEFAULT_PARAMETER_NORM_CEILING = 10.0
DEFAULT_OVERFIT_GAP_CEILING = 0.05


@dataclass(frozen=True)
class QCNNStabilityResult:
    """Ranked stability experiments, histories, and selected configuration."""

    results: pd.DataFrame
    optimization_history: pd.DataFrame
    best_config: dict[str, object]


def evaluate_qcnn_stability(
    data: QMLTrainValidation,
    *,
    initialization_scales: list[float],
    learning_rates: list[float],
    train_sample_sizes: list[int],
    max_iter: int = 20,
    batch_size: int = 32,
    perturbation: float = 0.1,
    l2: float = 0.001,
    random_state: int = 42,
) -> QCNNStabilityResult:
    """Evaluate initialization, learning-rate, and sample-size combinations."""
    _validate_grid(
        data,
        initialization_scales=initialization_scales,
        learning_rates=learning_rates,
        train_sample_sizes=train_sample_sizes,
    )
    rows = []
    histories = []
    configurations = product(
        initialization_scales,
        learning_rates,
        train_sample_sizes,
    )
    for config_index, (initialization_scale, learning_rate, sample_size) in enumerate(
        configurations
    ):
        config_id = f"qcnn_stability_{config_index:03d}"
        sampled_data = _with_sampled_train(
            data,
            sample_size=sample_size,
            random_state=random_state,
        )
        result = train_qcnn(
            sampled_data,
            model_name=config_id,
            max_iter=max_iter,
            learning_rate=learning_rate,
            perturbation=perturbation,
            batch_size=batch_size,
            l2=l2,
            initialization_scale=initialization_scale,
            random_state=random_state,
        )
        row = _stability_row(
            result,
            config_id=config_id,
            initialization_scale=initialization_scale,
            learning_rate=learning_rate,
            train_sample_size=sample_size,
        )
        rows.append(row)
        history = result.training_loss.copy()
        history.insert(0, "config_id", config_id)
        history["initialization_scale"] = initialization_scale
        history["learning_rate"] = learning_rate
        history["train_sample_size"] = sample_size
        histories.append(history)

    results = (
        pd.DataFrame(rows)
        .sort_values(
            ["stable", "validation_log_loss", "loss_volatility", "config_id"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return QCNNStabilityResult(
        results=results,
        optimization_history=pd.concat(histories, ignore_index=True),
        best_config=_best_config(results.iloc[0]),
    )


def render_qcnn_stability_report(result: QCNNStabilityResult) -> str:
    """Document the selected configuration and observed failure modes."""
    best = result.best_config
    failure_counts = (
        result.results["failure_modes"]
        .replace("none", pd.NA)
        .dropna()
        .str.get_dummies(sep=",")
        .sum()
        .sort_values(ascending=False)
    )
    lines = [
        "# QCNN Training Stability",
        "",
        "Configurations are ranked with stable runs first, then by validation log "
        "loss and mini-batch loss volatility.",
        "",
        "## Selected stable configuration",
        "",
        f"- Configuration: `{best['config_id']}`",
        f"- Initialization scale: `{best['initialization_scale']}`",
        f"- Learning rate: `{best['learning_rate']}`",
        f"- Training sample rows: `{best['train_sample_size']}`",
        f"- Validation log loss: `{best['validation_log_loss']:.6f}`",
        f"- Median gradient norm: `{best['median_gradient_norm']:.6f}`",
        f"- Loss volatility: `{best['loss_volatility']:.6f}`",
        f"- Failure modes: `{best['failure_modes']}`",
        "",
        "## Known failure modes",
        "",
    ]
    if failure_counts.empty:
        lines.append("No configured failure threshold was crossed in this grid.")
    else:
        for name, count in failure_counts.items():
            lines.append(f"- `{name}`: {int(count)} configuration(s)")
    lines.extend(["", "## Results", "", _markdown_table(result.results), ""])
    return "\n".join(lines)


def save_qcnn_stability_result(
    result: QCNNStabilityResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save ranked experiments, detailed histories, config, and report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": output_dir / "qcnn_stability_results.parquet",
        "optimization_history": output_dir / "qcnn_optimization_history.parquet",
        "best_config": output_dir / "qcnn_stable_config.json",
        "report": output_dir / "qcnn_stability_report.md",
    }
    result.results.to_parquet(paths["results"], index=False)
    result.optimization_history.to_parquet(
        paths["optimization_history"],
        index=False,
    )
    paths["best_config"].write_text(
        json.dumps(result.best_config, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        render_qcnn_stability_report(result),
        encoding="utf-8",
    )
    return paths


def _stability_row(
    result: QCNNResult,
    *,
    config_id: str,
    initialization_scale: float,
    learning_rate: float,
    train_sample_size: int,
) -> dict[str, object]:
    history = result.training_loss
    loss_differences = np.diff(history["loss"].to_numpy(dtype=float))
    loss_volatility = float(np.std(loss_differences)) if len(loss_differences) else 0.0
    median_gradient = float(history["gradient_norm"].median())
    maximum_gradient = float(history["gradient_norm"].max())
    maximum_parameter_norm = float(history["parameter_norm"].max())
    train_loss = float(result.training_metrics.iloc[0]["log_loss"])
    validation_loss = float(result.validation_metrics.iloc[0]["log_loss"])
    failure_modes = []
    numeric = history[["loss", "gradient_norm", "parameter_norm"]].to_numpy()
    if not np.isfinite(numeric).all():
        failure_modes.append("non_finite")
    if median_gradient < DEFAULT_GRADIENT_FLOOR:
        failure_modes.append("vanishing_gradient")
    if maximum_gradient > DEFAULT_GRADIENT_CEILING:
        failure_modes.append("exploding_gradient")
    if loss_volatility > DEFAULT_LOSS_VOLATILITY_CEILING:
        failure_modes.append("unstable_loss")
    if maximum_parameter_norm > DEFAULT_PARAMETER_NORM_CEILING:
        failure_modes.append("parameter_growth")
    if validation_loss - train_loss > DEFAULT_OVERFIT_GAP_CEILING:
        failure_modes.append("overfitting")
    return {
        "config_id": config_id,
        "initialization_scale": initialization_scale,
        "learning_rate": learning_rate,
        "train_sample_size": train_sample_size,
        "max_iter": int(result.config.params["max_iter"]),
        "initial_batch_loss": float(history.iloc[0]["loss"]),
        "final_batch_loss": float(history.iloc[-1]["loss"]),
        "loss_volatility": loss_volatility,
        "median_gradient_norm": median_gradient,
        "maximum_gradient_norm": maximum_gradient,
        "maximum_parameter_norm": maximum_parameter_norm,
        "train_log_loss": train_loss,
        "validation_log_loss": validation_loss,
        "overfit_gap": validation_loss - train_loss,
        "train_accuracy": float(result.training_metrics.iloc[0]["accuracy"]),
        "validation_accuracy": float(result.validation_metrics.iloc[0]["accuracy"]),
        "stable": not failure_modes,
        "failure_modes": ",".join(failure_modes) if failure_modes else "none",
    }


def _with_sampled_train(
    data: QMLTrainValidation,
    *,
    sample_size: int,
    random_state: int,
) -> QMLTrainValidation:
    targets = data.train.y.astype(int)
    per_class = sample_size // 2
    selected = []
    for target, indices in targets.groupby(targets).groups.items():
        if len(indices) < per_class:
            raise ValueError(
                f"Training sample does not contain {per_class} rows for class {target}."
            )
        rng = np.random.default_rng(random_state + int(target))
        selected.extend(
            rng.choice(list(indices), size=per_class, replace=False).tolist()
        )
    selected = sorted(selected)
    train = QMLDataset(
        X=data.train.X.iloc[selected].reset_index(drop=True),
        y=data.train.y.iloc[selected].reset_index(drop=True),
        metadata=data.train.metadata.iloc[selected].reset_index(drop=True),
    )
    return QMLTrainValidation(
        train=train,
        validation=data.validation,
        feature_columns=data.feature_columns,
        split_id=data.split_id,
    )


def _validate_grid(
    data: QMLTrainValidation,
    *,
    initialization_scales: list[float],
    learning_rates: list[float],
    train_sample_sizes: list[int],
) -> None:
    if not initialization_scales or any(value <= 0 for value in initialization_scales):
        raise ValueError("initialization_scales must contain positive values.")
    if not learning_rates or any(value <= 0 for value in learning_rates):
        raise ValueError("learning_rates must contain positive values.")
    if not train_sample_sizes or any(
        value <= 1 or value % 2 for value in train_sample_sizes
    ):
        raise ValueError("train_sample_sizes must contain positive even values.")
    if max(train_sample_sizes) > len(data.train.y):
        raise ValueError("train_sample_sizes cannot exceed available training rows.")


def _best_config(row: pd.Series) -> dict[str, object]:
    result = {}
    for key, value in row.items():
        if key == "rank":
            continue
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def _markdown_table(data: pd.DataFrame) -> str:
    columns = [
        "rank",
        "config_id",
        "initialization_scale",
        "learning_rate",
        "train_sample_size",
        "validation_log_loss",
        "median_gradient_norm",
        "loss_volatility",
        "stable",
        "failure_modes",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in data[columns].itertuples(index=False, name=None):
        values = [
            f"{value:.6f}" if isinstance(value, float) else str(value) for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
