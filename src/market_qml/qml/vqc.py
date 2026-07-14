"""Variational quantum classifier baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from market_qml.models.predictions import build_prediction_table
from market_qml.models.predictions import save_predictions as save_prediction_table
from market_qml.qml.encoding import AngleEncodingConfig, angle_encode_dataset
from market_qml.qml.interface import BaseQMLModel, QMLDataset, QMLModelConfig
from market_qml.qml.interface import QMLTrainValidation


MODEL_NAME = "vqc"
DEFAULT_N_QUBITS = 8
DEFAULT_ANSATZ_DEPTH = 1
DEFAULT_MAX_ITER = 100
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_L2 = 0.001
DEFAULT_PERTURBATION = 0.1
DEFAULT_BATCH_SIZE = 32
DEFAULT_OPTIMIZER = "spsa"
SUPPORTED_OPTIMIZERS = {"spsa", "finite_difference"}
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path("artifacts/models/vqc.pkl")
DEFAULT_PREDICTION_PATH = Path("data/processed/predictions_vqc.parquet")
DEFAULT_TRAINING_LOSS_PATH = Path("data/processed/vqc_training_loss.parquet")
DEFAULT_VALIDATION_METRICS_PATH = Path("data/processed/vqc_validation_metrics.parquet")


@dataclass(frozen=True)
class VQCResult:
    """Fitted VQC baseline, predictions, loss curve, and validation metrics."""

    model: "VariationalQuantumClassifier"
    predictions: pd.DataFrame
    training_loss: pd.DataFrame
    validation_metrics: pd.DataFrame
    config: QMLModelConfig


class VariationalQuantumClassifier(BaseQMLModel):
    """Binary VQC executed by an exact local statevector simulator.

    Each sample is encoded with one RY rotation per qubit. The variational
    ansatz alternates trainable RY rotations with a ring of CNOT entanglers, and
    the positive-class score is the probability of measuring qubit zero as 1.
    Circuit parameters are trained with reproducible SPSA updates.
    """

    def __init__(self, config: QMLModelConfig) -> None:
        super().__init__(config)
        self.n_qubits = int(config.params.get("n_qubits", DEFAULT_N_QUBITS))
        self.ansatz_depth = int(
            config.params.get("ansatz_depth", DEFAULT_ANSATZ_DEPTH)
        )
        self.max_iter = int(config.params.get("max_iter", DEFAULT_MAX_ITER))
        self.learning_rate = float(
            config.params.get("learning_rate", DEFAULT_LEARNING_RATE)
        )
        self.l2 = float(config.params.get("l2", DEFAULT_L2))
        self.perturbation = float(
            config.params.get("perturbation", DEFAULT_PERTURBATION)
        )
        self.batch_size = int(config.params.get("batch_size", DEFAULT_BATCH_SIZE))
        self.optimizer = str(config.params.get("optimizer", DEFAULT_OPTIMIZER))
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, dataset: QMLDataset) -> "VariationalQuantumClassifier":
        """Fit trainable variational readout weights on binary labels."""
        self._validate_hyperparameters()
        y = _binary_targets(dataset.y, require_two_classes=True)
        angles = _encoded_angles(dataset, n_qubits=self.n_qubits)

        rng = np.random.default_rng(self.seed)
        weights = rng.uniform(
            low=-0.1,
            high=0.1,
            size=(self.ansatz_depth, self.n_qubits),
        )
        losses = []
        for _ in range(self.max_iter):
            batch_indices = rng.choice(
                len(y),
                size=min(self.batch_size, len(y)),
                replace=False,
            )
            batch_angles = angles[batch_indices]
            batch_targets = y[batch_indices]
            gradient = _estimate_gradient(
                optimizer=self.optimizer,
                angles=batch_angles,
                targets=batch_targets,
                weights=weights,
                l2=self.l2,
                perturbation=self.perturbation,
                rng=rng,
            )
            weights -= self.learning_rate * gradient
            # Record a coherent post-update objective on the same mini-batch.
            losses.append(
                _circuit_loss(
                    batch_angles,
                    batch_targets,
                    weights,
                    self.l2,
                )
            )

        self.weights_ = weights
        self.loss_history_ = losses
        return self

    def predict_scores(self, dataset: QMLDataset) -> list[float]:
        """Predict positive-class probabilities for validation rows."""
        if self.weights_ is None:
            raise ValueError("VQC model must be fitted before prediction.")
        angles = _encoded_angles(dataset, n_qubits=self.n_qubits)
        return _circuit_probabilities(angles, self.weights_).tolist()

    def _validate_hyperparameters(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")
        if self.ansatz_depth <= 0:
            raise ValueError("ansatz_depth must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative.")
        if self.perturbation <= 0:
            raise ValueError("perturbation must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.optimizer not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                "optimizer must be one of: "
                + ", ".join(sorted(SUPPORTED_OPTIMIZERS))
            )


def train_vqc(
    data: QMLTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int | None = None,
    n_qubits: int = DEFAULT_N_QUBITS,
    ansatz_depth: int = DEFAULT_ANSATZ_DEPTH,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    l2: float = DEFAULT_L2,
    perturbation: float = DEFAULT_PERTURBATION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimizer: str = DEFAULT_OPTIMIZER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> VQCResult:
    """Train a VQC baseline on one QML train/validation split."""
    split_id = data.split_id if split_id is None else split_id
    config = QMLModelConfig(
        model_name=model_name,
        seed=random_state,
        params={
            "n_qubits": n_qubits,
            "ansatz_depth": ansatz_depth,
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
    model = VariationalQuantumClassifier(config)
    model.fit(data.train)
    y_score = model.predict_scores(data.validation)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
        split_id=split_id,
    )
    training_loss = build_training_loss_table(
        model.loss_history_,
        model_name=model_name,
        split_id=split_id,
    )
    validation_metrics = build_vqc_validation_metrics(
        predictions,
        model_name=model_name,
        split_id=split_id,
    )
    return VQCResult(
        model=model,
        predictions=predictions,
        training_loss=training_loss,
        validation_metrics=validation_metrics,
        config=config,
    )


def build_training_loss_table(
    losses: list[float],
    *,
    model_name: str,
    split_id: int,
) -> pd.DataFrame:
    """Build a loss curve table for VQC training diagnostics."""
    return pd.DataFrame(
        {
            "model_name": model_name,
            "split_id": split_id,
            "iteration": list(range(1, len(losses) + 1)),
            "loss": losses,
        }
    )


def build_vqc_validation_metrics(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split_id: int,
) -> pd.DataFrame:
    """Build compact validation diagnostics for one VQC split."""
    y_true = _binary_targets(predictions["y_true"], require_two_classes=False)
    y_score = pd.to_numeric(predictions["y_score"], errors="coerce").to_numpy()
    if np.isnan(y_score).any():
        raise ValueError("VQC validation scores contain missing values.")
    y_pred = (y_score >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "model_name": [model_name],
            "split_id": [split_id],
            "log_loss": [_binary_cross_entropy(y_true, y_score)],
            "accuracy": [float((y_pred == y_true).mean())],
            "brier_score": [float(np.mean((y_score - y_true) ** 2))],
        }
    )


def save_vqc_model(
    model: VariationalQuantumClassifier,
    output_path: str | Path = DEFAULT_MODEL_PATH,
) -> None:
    """Save a fitted VQC baseline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(model, f)


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path = DEFAULT_PREDICTION_PATH,
) -> None:
    """Save VQC predictions to parquet."""
    save_prediction_table(predictions, output_path)


def save_training_loss(
    training_loss: pd.DataFrame,
    output_path: str | Path = DEFAULT_TRAINING_LOSS_PATH,
) -> None:
    """Save VQC training loss diagnostics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_loss.to_parquet(output_path, index=False)


def save_validation_metrics(
    validation_metrics: pd.DataFrame,
    output_path: str | Path = DEFAULT_VALIDATION_METRICS_PATH,
) -> None:
    """Save VQC validation diagnostics to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_parquet(output_path, index=False)


def _encoded_angles(dataset: QMLDataset, *, n_qubits: int) -> np.ndarray:
    encoded = angle_encode_dataset(
        dataset,
        config=AngleEncodingConfig(n_qubits=n_qubits),
    )
    return encoded.X.to_numpy(dtype=float)


def _circuit_loss(
    angles: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    scores = _circuit_probabilities(angles, weights)
    return _binary_cross_entropy(targets, scores) + _l2_penalty(weights, l2)


def _estimate_gradient(
    *,
    optimizer: str,
    angles: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    l2: float,
    perturbation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate the circuit-objective gradient for a supported optimizer."""
    if optimizer == "spsa":
        direction = rng.choice((-1.0, 1.0), size=weights.shape)
        loss_plus = _circuit_loss(
            angles,
            targets,
            weights + perturbation * direction,
            l2,
        )
        loss_minus = _circuit_loss(
            angles,
            targets,
            weights - perturbation * direction,
            l2,
        )
        return (loss_plus - loss_minus) / (2.0 * perturbation) * direction

    if optimizer == "finite_difference":
        gradient = np.zeros_like(weights)
        for index in np.ndindex(weights.shape):
            offset = np.zeros_like(weights)
            offset[index] = perturbation
            loss_plus = _circuit_loss(
                angles,
                targets,
                weights + offset,
                l2,
            )
            loss_minus = _circuit_loss(
                angles,
                targets,
                weights - offset,
                l2,
            )
            gradient[index] = (loss_plus - loss_minus) / (2.0 * perturbation)
        return gradient

    raise ValueError(
        "optimizer must be one of: " + ", ".join(sorted(SUPPORTED_OPTIMIZERS))
    )


def _circuit_probabilities(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Execute the batched VQC circuit and return P(measure q0 = 1)."""
    if angles.ndim != 2:
        raise ValueError("VQC angles must be a two-dimensional array.")
    depth, n_qubits = weights.shape
    if angles.shape[1] != n_qubits:
        raise ValueError("VQC angle and circuit qubit counts must match.")

    state = np.zeros((len(angles), 1 << n_qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    for qubit in range(n_qubits):
        state = _apply_ry(state, angles[:, qubit], qubit)
    for layer in range(depth):
        for qubit in range(n_qubits):
            state = _apply_ry(state, weights[layer, qubit], qubit)
        for control in range(n_qubits):
            state = _apply_cnot(state, control, (control + 1) % n_qubits)

    basis = np.arange(1 << n_qubits)
    measured_one = ((basis >> 0) & 1).astype(bool)
    return np.sum(np.abs(state[:, measured_one]) ** 2, axis=1).real


def _apply_ry(
    state: np.ndarray,
    angles: np.ndarray | float,
    qubit: int,
) -> np.ndarray:
    """Apply an RY gate to one qubit for every state in a batch."""
    result = state.copy()
    basis = np.arange(state.shape[1])
    zero_indices = basis[(basis & (1 << qubit)) == 0]
    one_indices = zero_indices | (1 << qubit)
    theta = np.asarray(angles, dtype=float)
    if theta.ndim == 0:
        theta = np.full(len(state), float(theta))
    cosine = np.cos(theta / 2.0)[:, None]
    sine = np.sin(theta / 2.0)[:, None]
    zero = state[:, zero_indices]
    one = state[:, one_indices]
    result[:, zero_indices] = cosine * zero - sine * one
    result[:, one_indices] = sine * zero + cosine * one
    return result


def _apply_cnot(
    state: np.ndarray,
    control: int,
    target: int,
) -> np.ndarray:
    """Apply a CNOT by permuting statevector basis amplitudes."""
    basis = np.arange(state.shape[1])
    permutation = basis.copy()
    control_on = (basis & (1 << control)) != 0
    permutation[control_on] ^= 1 << target
    return state[:, permutation]


def _binary_targets(y: pd.Series, *, require_two_classes: bool) -> np.ndarray:
    targets = pd.to_numeric(y, errors="coerce")
    if targets.isna().any():
        raise ValueError("VQC targets contain missing or non-numeric values.")
    values = targets.to_numpy(dtype=float)
    if not set(values.tolist()).issubset({0.0, 1.0}):
        raise ValueError("VQC requires binary targets encoded as 0/1.")
    if require_two_classes and len(set(values.tolist())) < 2:
        raise ValueError("VQC training labels must contain at least two classes.")
    return values


def _binary_cross_entropy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    eps = 1e-12
    scores = np.clip(y_score, eps, 1 - eps)
    return float(
        -np.mean(y_true * np.log(scores) + (1 - y_true) * np.log(1 - scores))
    )


def _l2_penalty(weights: np.ndarray, l2: float) -> float:
    return float(0.5 * l2 * np.sum(weights**2))
