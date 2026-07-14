"""Quantum fidelity-kernel support vector machine baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from market_qml.models.predictions import build_prediction_table, save_predictions
from market_qml.qml.feature_map import (
    QuantumFeatureMapConfig,
    QuantumKernelFeatureMap,
    fidelity_kernel,
)
from market_qml.qml.interface import BaseQMLModel, QMLDataset, QMLModelConfig
from market_qml.qml.interface import QMLTrainValidation


MODEL_NAME = "qsvm"
DEFAULT_C = 1.0
DEFAULT_N_QUBITS = 8
DEFAULT_REPETITIONS = 2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path("artifacts/models/qsvm.pkl")
DEFAULT_PREDICTION_PATH = Path("data/processed/predictions_qsvm.parquet")
DEFAULT_DIAGNOSTICS_PATH = Path("data/processed/qsvm_kernel_diagnostics.parquet")
DEFAULT_KERNEL_PATH = Path("artifacts/qml/qsvm/kernel_matrices.npz")


@dataclass(frozen=True)
class QSVMResult:
    """Fitted QSVM, standard predictions, kernels, and diagnostics."""

    model: "QuantumKernelSVM"
    predictions: pd.DataFrame
    train_kernel: np.ndarray
    validation_kernel: np.ndarray
    kernel_diagnostics: pd.DataFrame
    config: QMLModelConfig


class QuantumKernelSVM(BaseQMLModel):
    """SVM classifier trained on exact quantum-state fidelity kernels."""

    def __init__(self, config: QMLModelConfig) -> None:
        super().__init__(config)
        self.C = float(config.params.get("C", DEFAULT_C))
        self.n_qubits = int(config.params.get("n_qubits", DEFAULT_N_QUBITS))
        self.repetitions = int(
            config.params.get("repetitions", DEFAULT_REPETITIONS)
        )
        self.feature_map = QuantumKernelFeatureMap(
            QuantumFeatureMapConfig(
                n_qubits=self.n_qubits,
                repetitions=self.repetitions,
            )
        )
        self.estimator_: SVC | None = None
        self.train_states_: np.ndarray | None = None
        self.train_kernel_: np.ndarray | None = None
        self.last_prediction_kernel_: np.ndarray | None = None

    def fit(self, dataset: QMLDataset) -> "QuantumKernelSVM":
        """Build the train fidelity matrix and fit a precomputed-kernel SVM."""
        if self.C <= 0:
            raise ValueError("C must be positive.")
        y = _binary_targets(dataset.y, require_two_classes=True)
        self.train_states_ = self.feature_map.transform(dataset).states
        self.train_kernel_ = fidelity_kernel(
            self.train_states_,
            self.train_states_,
        )
        self.estimator_ = SVC(
            C=self.C,
            kernel="precomputed",
            probability=True,
            random_state=self.seed,
        )
        self.estimator_.fit(self.train_kernel_, y)
        return self

    def predict_scores(self, dataset: QMLDataset) -> list[float]:
        """Return calibrated positive-class scores from a cross-kernel matrix."""
        if self.estimator_ is None or self.train_states_ is None:
            raise ValueError("QSVM model must be fitted before prediction.")
        states = self.feature_map.transform(dataset).states
        self.last_prediction_kernel_ = fidelity_kernel(states, self.train_states_)
        positive_class_index = int(np.where(self.estimator_.classes_ == 1)[0][0])
        return self.estimator_.predict_proba(self.last_prediction_kernel_)[
            :, positive_class_index
        ].tolist()


def train_qsvm(
    data: QMLTrainValidation,
    *,
    model_name: str = MODEL_NAME,
    split_id: int | None = None,
    C: float = DEFAULT_C,
    n_qubits: int = DEFAULT_N_QUBITS,
    repetitions: int = DEFAULT_REPETITIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> QSVMResult:
    """Train a QSVM and produce standard validation predictions."""
    split_id = data.split_id if split_id is None else split_id
    config = QMLModelConfig(
        model_name=model_name,
        seed=random_state,
        params={
            "C": C,
            "n_qubits": n_qubits,
            "repetitions": repetitions,
            "kernel": "quantum_state_fidelity",
            "backend": "numpy_statevector",
            "probability_calibration": True,
        },
    )
    model = QuantumKernelSVM(config).fit(data.train)
    y_score = model.predict_scores(data.validation)
    predictions = build_prediction_table(
        metadata=data.validation.metadata,
        y_true=data.validation.y,
        y_score=y_score,
        model_name=model_name,
        split_id=split_id,
    )
    validation_kernel = model.last_prediction_kernel_
    if model.train_kernel_ is None or validation_kernel is None:
        raise RuntimeError("QSVM kernel matrices were not produced.")
    diagnostics = build_kernel_diagnostics(
        train_kernel=model.train_kernel_,
        validation_kernel=validation_kernel,
        model=model,
        split_id=split_id,
    )
    return QSVMResult(
        model=model,
        predictions=predictions,
        train_kernel=model.train_kernel_,
        validation_kernel=validation_kernel,
        kernel_diagnostics=diagnostics,
        config=config,
    )


def build_kernel_diagnostics(
    *,
    train_kernel: np.ndarray,
    validation_kernel: np.ndarray,
    model: QuantumKernelSVM,
    split_id: int,
) -> pd.DataFrame:
    """Summarize kernel dimensions, values, symmetry, and fitted SVM size."""
    rows = []
    for name, matrix in [
        ("train", train_kernel),
        ("validation", validation_kernel),
    ]:
        rows.append(
            {
                "model_name": model.model_name,
                "split_id": split_id,
                "matrix": name,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "minimum": float(matrix.min()),
                "maximum": float(matrix.max()),
                "mean": float(matrix.mean()),
                "symmetry_max_error": (
                    float(np.max(np.abs(matrix - matrix.T)))
                    if matrix.shape[0] == matrix.shape[1]
                    else np.nan
                ),
                "diagonal_minimum": (
                    float(np.diag(matrix).min())
                    if matrix.shape[0] == matrix.shape[1]
                    else np.nan
                ),
                "diagonal_maximum": (
                    float(np.diag(matrix).max())
                    if matrix.shape[0] == matrix.shape[1]
                    else np.nan
                ),
                "support_vectors": (
                    int(model.estimator_.support_.size)
                    if model.estimator_ is not None
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def save_qsvm_result(
    result: QSVMResult,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    prediction_path: str | Path = DEFAULT_PREDICTION_PATH,
    diagnostics_path: str | Path = DEFAULT_DIAGNOSTICS_PATH,
    kernel_path: str | Path = DEFAULT_KERNEL_PATH,
) -> dict[str, Path]:
    """Save the fitted model, predictions, diagnostics, and kernel matrices."""
    paths = {
        "model": Path(model_path),
        "predictions": Path(prediction_path),
        "kernel_diagnostics": Path(diagnostics_path),
        "kernel_matrices": Path(kernel_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths["model"].open("wb") as handle:
        pickle.dump(result.model, handle)
    save_predictions(result.predictions, paths["predictions"])
    result.kernel_diagnostics.to_parquet(paths["kernel_diagnostics"], index=False)
    np.savez_compressed(
        paths["kernel_matrices"],
        train_kernel=result.train_kernel,
        validation_kernel=result.validation_kernel,
    )
    return paths


def _binary_targets(y: pd.Series, *, require_two_classes: bool) -> np.ndarray:
    targets = pd.to_numeric(y, errors="coerce")
    if targets.isna().any() or not set(targets.tolist()).issubset({0, 1}):
        raise ValueError("QSVM requires binary targets encoded as 0/1.")
    values = targets.to_numpy(dtype=int)
    if require_two_classes and len(np.unique(values)) < 2:
        raise ValueError("QSVM training labels must contain at least two classes.")
    return values
