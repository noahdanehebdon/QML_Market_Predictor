"""Reusable convolution and pooling blocks for the eight-qubit QCNN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_qml.qml.simulator import apply_cnot, apply_ry

DEFAULT_N_QUBITS = 8
DEFAULT_RANDOM_STATE = 42
CONVOLUTION_PARAMETER_COUNT = 4
POOLING_PARAMETER_COUNT = 1


@dataclass(frozen=True)
class QCNNArchitecture:
    """Constructed QCNN blocks, parameter layout, and active-qubit flow."""

    n_qubits: int
    operations: pd.DataFrame
    active_qubit_flow: tuple[tuple[int, ...], ...]
    parameter_count: int
    block_parameter_shapes: dict[str, tuple[int, ...]]


def two_qubit_convolution(
    state: np.ndarray,
    first: int,
    second: int,
    parameters: np.ndarray,
) -> np.ndarray:
    """Apply a four-parameter bidirectional two-qubit convolution block."""
    values = _parameter_vector(
        parameters,
        expected=CONVOLUTION_PARAMETER_COUNT,
        block_name="convolution",
    )
    result = apply_ry(state, values[0], first)
    result = apply_ry(result, values[1], second)
    result = apply_cnot(result, first, second)
    result = apply_ry(result, values[2], second)
    result = apply_cnot(result, second, first)
    return apply_ry(result, values[3], first)


def pooling_block(
    state: np.ndarray,
    source: int,
    target: int,
    parameters: np.ndarray,
) -> np.ndarray:
    """Transfer information from a retiring source into an active target."""
    values = _parameter_vector(
        parameters,
        expected=POOLING_PARAMETER_COUNT,
        block_name="pooling",
    )
    result = apply_cnot(state, source, target)
    return apply_ry(result, values[0], target)


def build_qcnn_architecture(n_qubits: int = DEFAULT_N_QUBITS) -> QCNNArchitecture:
    """Construct the minimal pairwise QCNN reduction from 8 to 4 to 2 qubits."""
    if n_qubits != DEFAULT_N_QUBITS:
        raise ValueError("The minimal QCNN architecture currently requires 8 qubits.")

    rows: list[dict[str, object]] = []
    parameter_index = 0
    operation_index = 0
    active = tuple(range(n_qubits))
    active_flow = [active]

    for stage in range(2):
        pairs = list(zip(active[::2], active[1::2]))
        for block_index, (first, second) in enumerate(pairs):
            block_rows, parameter_index, operation_index = _convolution_operations(
                stage=stage,
                block_index=block_index,
                first=first,
                second=second,
                parameter_index=parameter_index,
                operation_index=operation_index,
            )
            rows.extend(block_rows)
        for block_index, (target, source) in enumerate(pairs):
            block_rows, parameter_index, operation_index = _pooling_operations(
                stage=stage,
                block_index=block_index,
                source=source,
                target=target,
                parameter_index=parameter_index,
                operation_index=operation_index,
            )
            rows.extend(block_rows)
        active = tuple(pair[0] for pair in pairs)
        active_flow.append(active)

    operations = pd.DataFrame(rows)
    operations["parameter_index"] = operations["parameter_index"].astype("Int64")
    return QCNNArchitecture(
        n_qubits=n_qubits,
        operations=operations,
        active_qubit_flow=tuple(active_flow),
        parameter_count=parameter_index,
        block_parameter_shapes={
            "convolution": (CONVOLUTION_PARAMETER_COUNT,),
            "pooling": (POOLING_PARAMETER_COUNT,),
            "complete_circuit": (parameter_index,),
        },
    )


def initialize_qcnn_parameters(
    architecture: QCNNArchitecture,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    scale: float = 0.1,
) -> np.ndarray:
    """Initialize a reproducible flat parameter vector for all QCNN blocks."""
    if scale <= 0:
        raise ValueError("scale must be positive.")
    rng = np.random.default_rng(random_state)
    return rng.uniform(
        low=-scale,
        high=scale,
        size=architecture.parameter_count,
    )


def execute_qcnn_architecture(
    state: np.ndarray,
    architecture: QCNNArchitecture,
    parameters: np.ndarray,
) -> np.ndarray:
    """Execute a constructed QCNN operation table on a statevector batch."""
    values = _parameter_vector(
        parameters,
        expected=architecture.parameter_count,
        block_name="complete QCNN",
    )
    result = state
    for operation in architecture.operations.itertuples(index=False):
        qubits = tuple(int(value) for value in str(operation.qubits).split(","))
        if operation.gate == "ry":
            result = apply_ry(
                result,
                values[int(operation.parameter_index)],
                qubits[0],
            )
        elif operation.gate == "cnot":
            result = apply_cnot(result, qubits[0], qubits[1])
        else:
            raise ValueError(f"Unsupported QCNN operation gate: {operation.gate}.")
    return result


def _convolution_operations(
    *,
    stage: int,
    block_index: int,
    first: int,
    second: int,
    parameter_index: int,
    operation_index: int,
) -> tuple[list[dict[str, object]], int, int]:
    specification = [
        ("ry", (first,), True),
        ("ry", (second,), True),
        ("cnot", (first, second), False),
        ("ry", (second,), True),
        ("cnot", (second, first), False),
        ("ry", (first,), True),
    ]
    return _operation_rows(
        specification,
        stage=stage,
        block="convolution",
        block_index=block_index,
        parameter_index=parameter_index,
        operation_index=operation_index,
    )


def _pooling_operations(
    *,
    stage: int,
    block_index: int,
    source: int,
    target: int,
    parameter_index: int,
    operation_index: int,
) -> tuple[list[dict[str, object]], int, int]:
    return _operation_rows(
        [("cnot", (source, target), False), ("ry", (target,), True)],
        stage=stage,
        block="pooling",
        block_index=block_index,
        parameter_index=parameter_index,
        operation_index=operation_index,
    )


def _operation_rows(
    specification: list[tuple[str, tuple[int, ...], bool]],
    *,
    stage: int,
    block: str,
    block_index: int,
    parameter_index: int,
    operation_index: int,
) -> tuple[list[dict[str, object]], int, int]:
    rows = []
    for gate, qubits, parameterized in specification:
        rows.append(
            {
                "operation_index": operation_index,
                "stage": stage,
                "block": block,
                "block_index": block_index,
                "gate": gate,
                "qubits": ",".join(str(qubit) for qubit in qubits),
                "parameter_index": parameter_index if parameterized else pd.NA,
            }
        )
        operation_index += 1
        if parameterized:
            parameter_index += 1
    return rows, parameter_index, operation_index


def _parameter_vector(
    parameters: np.ndarray,
    *,
    expected: int,
    block_name: str,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.shape != (expected,):
        raise ValueError(
            f"{block_name} parameters must have shape ({expected},), got "
            f"{values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{block_name} parameters must be finite.")
    return values
