"""Small batched exact-statevector backend for repository QML circuits."""

from __future__ import annotations

import numpy as np


BACKEND_NAME = "numpy_statevector"


def zero_state(batch_size: int, n_qubits: int) -> np.ndarray:
    """Return a batch of ``|0...0>`` statevectors."""
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    state = np.zeros((batch_size, 1 << n_qubits), dtype=np.complex128)
    state[:, 0] = 1.0
    return state


def apply_ry(
    state: np.ndarray,
    angles: np.ndarray | float,
    qubit: int,
) -> np.ndarray:
    """Apply an RY gate to one qubit for every state in a batch."""
    _validate_qubit(state, qubit)
    result = state.copy()
    basis = np.arange(state.shape[1])
    zero_indices = basis[(basis & (1 << qubit)) == 0]
    one_indices = zero_indices | (1 << qubit)
    theta = np.asarray(angles, dtype=float)
    if theta.ndim == 0:
        theta = np.full(len(state), float(theta))
    if theta.shape != (len(state),):
        raise ValueError("RY angles must be scalar or contain one value per state.")
    cosine = np.cos(theta / 2.0)[:, None]
    sine = np.sin(theta / 2.0)[:, None]
    zero = state[:, zero_indices]
    one = state[:, one_indices]
    result[:, zero_indices] = cosine * zero - sine * one
    result[:, one_indices] = sine * zero + cosine * one
    return result


def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply a CNOT by permuting statevector basis amplitudes."""
    _validate_two_qubits(state, control, target)
    basis = np.arange(state.shape[1])
    permutation = basis.copy()
    control_on = (basis & (1 << control)) != 0
    permutation[control_on] ^= 1 << target
    return state[:, permutation]


def apply_cz(state: np.ndarray, first: int, second: int) -> np.ndarray:
    """Apply a controlled-Z phase to a pair of qubits."""
    _validate_two_qubits(state, first, second)
    basis = np.arange(state.shape[1])
    both_on = ((basis & (1 << first)) != 0) & ((basis & (1 << second)) != 0)
    result = state.copy()
    result[:, both_on] *= -1.0
    return result


def expectation_z(state: np.ndarray, qubit: int) -> np.ndarray:
    """Return the Pauli-Z expectation of one qubit for every batched state."""
    _validate_qubit(state, qubit)
    basis = np.arange(state.shape[1])
    signs = np.where((basis & (1 << qubit)) == 0, 1.0, -1.0)
    return np.sum(np.abs(state) ** 2 * signs[None, :], axis=1).real


def _validate_two_qubits(state: np.ndarray, first: int, second: int) -> None:
    _validate_qubit(state, first)
    _validate_qubit(state, second)
    if first == second:
        raise ValueError("Two-qubit gates require distinct qubits.")


def _validate_qubit(state: np.ndarray, qubit: int) -> None:
    if state.ndim != 2 or state.shape[1] <= 0:
        raise ValueError("State must be a two-dimensional statevector batch.")
    n_qubits = int(np.log2(state.shape[1]))
    if 1 << n_qubits != state.shape[1]:
        raise ValueError("Statevector width must be a power of two.")
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"qubit must be between 0 and {n_qubits - 1}.")
