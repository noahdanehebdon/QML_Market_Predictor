"""Trainable eight-qubit quantum convolutional neural network classifier."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.models.predictions import build_prediction_table, save_predictions
from market_qml.qml.encoding import AngleEncodingConfig, angle_encode_dataset
from market_qml.qml.interface import (
    BaseQMLModel,
    QMLDataset,
    QMLModelConfig,
    QMLTrainValidation,
)
from market_qml.qml.qcnn_blocks import (
    QCNNArchitecture,
    build_qcnn_architecture,
    execute_qcnn_architecture,
    initialize_qcnn_parameters,
)
from market_qml.qml.simulator import apply_ry, expectation_z, zero_state

MODEL_NAME = "qcnn"
DEFAULT_N_QUBITS = 8
DEFAULT_MAX_ITER = 50
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_PERTURBATION = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_L2 = 0.001
DEFAULT_INITIALIZATION_SCALE = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_OUTPUT_DIR = Path("artifacts/qml/qcnn")


@dataclass(frozen=True)
class QCNNResult:
    """Fitted QCNN with predictions, loss curve, and split diagnostics."""

    model: QuantumConvolutionalClassifier
    predictions: pd.DataFrame
    training_loss: pd.DataFrame
    training_metrics: pd.DataFrame
    validation_metrics: pd.DataFrame
    config: QMLModelConfig


class QuantumConvolutionalClassifier(BaseQMLModel):
    """Eight-qubit angle-encoded QCNN trained with SPSA."""

    def __init__(self, config: QMLModelConfig) -> None:
        super().__init__(config)
        self.n_qubits = int(config.params.get("n_qubits", DEFAULT_N_QUBITS))
        self.max_iter = int(config.params.get("max_iter", DEFAULT_MAX_ITER))
        self.learning_rate = float(
            config.params.get("learning_rate", DEFAULT_LEARNING_RATE)
        )
        self.perturbation = float(
            config.params.get("perturbation", DEFAULT_PERTURBATION)
        )
        self.batch_size = int(config.params.get("batch_size", DEFAULT_BATCH_SIZE))
        self.l2 = float(config.params.get("l2", DEFAULT_L2))
        self.initialization_scale = float(
            config.params.get("initialization_scale", DEFAULT_INITIALIZATION_SCALE)
        )
        self.architecture: QCNNArchitecture = build_qcnn_architecture(self.n_qubits)
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.optimization_history_: list[dict[str, float | int]] = []

    def fit(self, dataset: QMLDataset) -> QuantumConvolutionalClassifier:
        """Fit the 30 QCNN parameters on binary labels with SPSA updates."""
        self._validate_hyperparameters()
        targets = _binary_targets(dataset.y, require_two_classes=True)
        angles = _encoded_angles(dataset, n_qubits=self.n_qubits)
        weights = initialize_qcnn_parameters(
            self.architecture,
            random_state=self.seed,
            scale=self.initialization_scale,
        )
        rng = np.random.default_rng(self.seed + 1)
        losses = []
        optimization_history = []
        for iteration in range(1, self.max_iter + 1):
            indices = rng.choice(
                len(targets),
                size=min(self.batch_size, len(targets)),
                replace=False,
            )
            batch_angles = angles[indices]
            batch_targets = targets[indices]
            direction = rng.choice((-1.0, 1.0), size=weights.shape)
            loss_plus = _qcnn_loss(
                batch_angles,
                batch_targets,
                weights + self.perturbation * direction,
                self.architecture,
                self.l2,
            )
            loss_minus = _qcnn_loss(
                batch_angles,
                batch_targets,
                weights - self.perturbation * direction,
                self.architecture,
                self.l2,
            )
            gradient = (loss_plus - loss_minus) / (2.0 * self.perturbation) * direction
            weights -= self.learning_rate * gradient
            post_update_loss = _qcnn_loss(
                batch_angles,
                batch_targets,
                weights,
                self.architecture,
                self.l2,
            )
            losses.append(post_update_loss)
            optimization_history.append(
                {
                    "iteration": iteration,
                    "loss": post_update_loss,
                    "gradient_norm": float(np.linalg.norm(gradient)),
                    "step_norm": float(self.learning_rate * np.linalg.norm(gradient)),
                    "parameter_norm": float(np.linalg.norm(weights)),
                    "batch_rows": len(indices),
                }
            )

        self.weights_ = weights
        self.loss_history_ = losses
        self.optimization_history_ = optimization_history
        return self

    def predict_scores(self, dataset: QMLDataset) -> list[float]:
        """Map the mean final-qubit Z expectation to a positive-class score."""
        if self.weights_ is None:
            raise ValueError("QCNN model must be fitted before prediction.")
        angles = _encoded_angles(dataset, n_qubits=self.n_qubits)
        return _qcnn_probabilities(
            angles,
            self.weights_,
            self.architecture,
        ).tolist()

    def _validate_hyperparameters(self) -> None:
        if self.n_qubits != DEFAULT_N_QUBITS:
            raise ValueError("QCNN currently requires exactly 8 qubits.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.perturbation <= 0:
            raise ValueError("perturbation must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative.")
        if self.initialization_scale <= 0:
            raise ValueError("initialization_scale must be positive.")


def train_qcnn(
    data: QMLTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    perturbation: float = DEFAULT_PERTURBATION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    l2: float = DEFAULT_L2,
    initialization_scale: float = DEFAULT_INITIALIZATION_SCALE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> QCNNResult:
    """Train the QCNN and build standard validation outputs."""
    split_id = data.split_id if split_id is None else split_id
    config = QMLModelConfig(
        model_name=model_name,
        seed=random_state,
        params={
            "n_qubits": DEFAULT_N_QUBITS,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "perturbation": perturbation,
            "batch_size": batch_size,
            "l2": l2,
            "initialization_scale": initialization_scale,
            "optimizer": "spsa",
            "encoding": "ry_angle",
            "readout_qubits": [0, 4],
            "backend": "numpy_statevector",
        },
    )
    model = QuantumConvolutionalClassifier(config).fit(data.train)
    validation_scores = model.predict_scores(data.validation)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=validation_scores,
        model_name=model_name,
        split_id=split_id,
    )
    train_scores = model.predict_scores(data.train)
    training_metrics = build_qcnn_metrics(
        data.train.y,
        train_scores,
        model_name=model_name,
        split_id=split_id,
        sample_role="train",
    )
    validation_metrics = build_qcnn_metrics(
        data.validation.y,
        validation_scores,
        model_name=model_name,
        split_id=split_id,
        sample_role="validation",
    )
    training_loss = pd.DataFrame(model.optimization_history_)
    training_loss.insert(0, "split_id", split_id)
    training_loss.insert(0, "model_name", model_name)
    return QCNNResult(
        model=model,
        predictions=predictions,
        training_loss=training_loss,
        training_metrics=training_metrics,
        validation_metrics=validation_metrics,
        config=config,
    )


def build_qcnn_metrics(
    targets: pd.Series,
    scores: list[float],
    *,
    model_name: str,
    split_id: int,
    sample_role: str,
) -> pd.DataFrame:
    """Build compact binary metrics for a QCNN split role."""
    y_true = _binary_targets(targets, require_two_classes=False)
    y_score = np.asarray(scores, dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "model_name": [model_name],
            "split_id": [split_id],
            "sample_role": [sample_role],
            "rows": [len(y_true)],
            "log_loss": [_binary_cross_entropy(y_true, y_score)],
            "accuracy": [float((y_pred == y_true).mean())],
            "brier_score": [float(np.mean((y_score - y_true) ** 2))],
        }
    )


def save_qcnn_result(
    result: QCNNResult,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Save the fitted QCNN, predictions, loss curve, and role metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": output_dir / "qcnn.pkl",
        "predictions": output_dir / "predictions.parquet",
        "training_loss": output_dir / "training_loss.parquet",
        "training_metrics": output_dir / "training_metrics.parquet",
        "validation_metrics": output_dir / "validation_metrics.parquet",
    }
    with paths["model"].open("wb") as handle:
        pickle.dump(result.model, handle)
    save_predictions(result.predictions, paths["predictions"])
    result.training_loss.to_parquet(paths["training_loss"], index=False)
    result.training_metrics.to_parquet(paths["training_metrics"], index=False)
    result.validation_metrics.to_parquet(paths["validation_metrics"], index=False)
    return paths


def _encoded_angles(dataset: QMLDataset, *, n_qubits: int) -> np.ndarray:
    encoded = angle_encode_dataset(
        dataset,
        config=AngleEncodingConfig(n_qubits=n_qubits),
        feature_columns=list(dataset.X.columns),
    )
    return encoded.X.to_numpy(dtype=float)


def _qcnn_probabilities(
    angles: np.ndarray,
    weights: np.ndarray,
    architecture: QCNNArchitecture,
) -> np.ndarray:
    state = zero_state(len(angles), architecture.n_qubits)
    for qubit in range(architecture.n_qubits):
        state = apply_ry(state, angles[:, qubit], qubit)
    state = execute_qcnn_architecture(state, architecture, weights)
    readout_qubits = architecture.active_qubit_flow[-1]
    mean_expectation = np.mean(
        [expectation_z(state, qubit) for qubit in readout_qubits],
        axis=0,
    )
    return np.clip((1.0 - mean_expectation) / 2.0, 0.0, 1.0)


def _qcnn_loss(
    angles: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    architecture: QCNNArchitecture,
    l2: float,
) -> float:
    scores = _qcnn_probabilities(angles, weights, architecture)
    return _binary_cross_entropy(targets, scores) + float(0.5 * l2 * np.sum(weights**2))


def _binary_targets(y: pd.Series, *, require_two_classes: bool) -> np.ndarray:
    targets = pd.to_numeric(y, errors="coerce")
    if targets.isna().any() or not set(targets.tolist()).issubset({0, 1}):
        raise ValueError("QCNN requires binary targets encoded as 0/1.")
    values = targets.to_numpy(dtype=float)
    if require_two_classes and len(np.unique(values)) < 2:
        raise ValueError("QCNN training labels must contain at least two classes.")
    return values


def _binary_cross_entropy(targets: np.ndarray, scores: np.ndarray) -> float:
    clipped = np.clip(scores, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    )
