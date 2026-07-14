"""Quantum feature map and fidelity-kernel primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_qml.qml.encoding import AngleEncodingConfig, angle_encode_dataset
from market_qml.qml.interface import QMLDataset, QMLTrainValidation
from market_qml.qml.simulator import BACKEND_NAME, apply_cz, apply_ry, zero_state


DEFAULT_N_QUBITS = 8
DEFAULT_REPETITIONS = 2


@dataclass(frozen=True)
class QuantumFeatureMapConfig:
    """Configuration for the angle-encoded, ring-entangled feature map."""

    n_qubits: int = DEFAULT_N_QUBITS
    repetitions: int = DEFAULT_REPETITIONS
    backend: str = BACKEND_NAME
    entanglement: str = "ring"


@dataclass(frozen=True)
class QuantumFeatureMapResult:
    """Encoded angles, normalized circuit states, and circuit description."""

    states: np.ndarray
    angles: pd.DataFrame
    metadata: pd.DataFrame
    operations: pd.DataFrame
    config: QuantumFeatureMapConfig


@dataclass(frozen=True)
class QuantumFeatureMapSplitResult:
    """Feature-map circuit outputs for one train/validation split."""

    train: QuantumFeatureMapResult
    validation: QuantumFeatureMapResult
    split_id: int


class QuantumKernelFeatureMap:
    """Map PCA rows to quantum states for fidelity-kernel estimation."""

    def __init__(self, config: QuantumFeatureMapConfig | None = None) -> None:
        self.config = config or QuantumFeatureMapConfig()
        _validate_config(self.config)

    def transform(self, dataset: QMLDataset) -> QuantumFeatureMapResult:
        """Execute the feature-map circuit for every row in a QML dataset."""
        encoded = angle_encode_dataset(
            dataset,
            config=AngleEncodingConfig(n_qubits=self.config.n_qubits),
        )
        angles = encoded.X.to_numpy(dtype=float)
        states = zero_state(len(angles), self.config.n_qubits)
        for _ in range(self.config.repetitions):
            for qubit in range(self.config.n_qubits):
                states = apply_ry(states, angles[:, qubit], qubit)
            for first, second in _ring_edges(self.config.n_qubits):
                states = apply_cz(states, first, second)

        return QuantumFeatureMapResult(
            states=states,
            angles=encoded.X,
            metadata=encoded.metadata,
            operations=feature_map_operations(self.config),
            config=self.config,
        )

    def transform_train_validation(
        self,
        data: QMLTrainValidation,
    ) -> QuantumFeatureMapSplitResult:
        """Apply the identical feature map to training and validation rows."""
        return QuantumFeatureMapSplitResult(
            train=self.transform(data.train),
            validation=self.transform(data.validation),
            split_id=data.split_id,
        )

    def kernel(
        self,
        left: QMLDataset,
        right: QMLDataset | None = None,
    ) -> np.ndarray:
        """Compute a fidelity kernel from feature-map state overlaps."""
        left_states = self.transform(left).states
        right_states = left_states if right is None else self.transform(right).states
        return fidelity_kernel(left_states, right_states)


def fidelity_kernel(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    """Return ``|<left|right>|^2`` for every pair of circuit states."""
    _validate_states(left_states, "left_states")
    _validate_states(right_states, "right_states")
    if left_states.shape[1] != right_states.shape[1]:
        raise ValueError("Feature-map statevector widths must match.")
    overlaps = left_states.conj() @ right_states.T
    kernel = np.abs(overlaps) ** 2
    return np.clip(kernel.real, 0.0, 1.0)


def feature_map_operations(config: QuantumFeatureMapConfig) -> pd.DataFrame:
    """Describe the feature rotations and ring entanglers in execution order."""
    rows: list[dict[str, object]] = []
    operation_index = 0
    for repetition in range(config.repetitions):
        for qubit in range(config.n_qubits):
            rows.append(
                {
                    "operation_index": operation_index,
                    "repetition": repetition,
                    "gate": "ry",
                    "qubits": str(qubit),
                    "parameter": f"theta_{qubit:02d}",
                }
            )
            operation_index += 1
        for first, second in _ring_edges(config.n_qubits):
            rows.append(
                {
                    "operation_index": operation_index,
                    "repetition": repetition,
                    "gate": "cz",
                    "qubits": f"{first},{second}",
                    "parameter": None,
                }
            )
            operation_index += 1
    return pd.DataFrame(rows)


def save_feature_map_split(
    result: QuantumFeatureMapSplitResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save circuit states and supporting metadata for kernel estimation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "states": output_dir / f"feature_map_states_split_{result.split_id:03d}.npz",
        "train_metadata": output_dir
        / f"feature_map_train_metadata_split_{result.split_id:03d}.parquet",
        "validation_metadata": output_dir
        / f"feature_map_validation_metadata_split_{result.split_id:03d}.parquet",
        "operations": output_dir
        / f"feature_map_operations_split_{result.split_id:03d}.parquet",
    }
    np.savez_compressed(
        paths["states"],
        train_states=result.train.states,
        validation_states=result.validation.states,
    )
    result.train.metadata.to_parquet(paths["train_metadata"], index=False)
    result.validation.metadata.to_parquet(paths["validation_metadata"], index=False)
    result.train.operations.to_parquet(paths["operations"], index=False)
    return paths


def _validate_config(config: QuantumFeatureMapConfig) -> None:
    if config.n_qubits < 2:
        raise ValueError("n_qubits must be at least 2 for ring entanglement.")
    if config.repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if config.backend != BACKEND_NAME:
        raise ValueError(f"Unsupported quantum feature-map backend: {config.backend}.")
    if config.entanglement != "ring":
        raise ValueError("entanglement must be 'ring'.")


def _ring_edges(n_qubits: int) -> list[tuple[int, int]]:
    edges = [(qubit, qubit + 1) for qubit in range(n_qubits - 1)]
    if n_qubits > 2:
        edges.append((n_qubits - 1, 0))
    return edges


def _validate_states(states: np.ndarray, name: str) -> None:
    if states.ndim != 2 or states.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty-width statevector matrix.")
    norms = np.sum(np.abs(states) ** 2, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-10):
        raise ValueError(f"{name} contains non-normalized quantum states.")
