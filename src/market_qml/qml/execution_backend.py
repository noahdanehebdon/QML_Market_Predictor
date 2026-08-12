"""Selectable exact, shot-based, noisy, and IBM hardware VQC execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_qml.qml.ibm_backend import IBMBackendConfig, run_ibm_vqc
from market_qml.qml.vqc import _circuit_probabilities

SUPPORTED_EXECUTION_MODES = {
    "exact",
    "shot_simulator",
    "noisy_simulator",
    "ibm_hardware",
}


@dataclass(frozen=True)
class VQCExecutionResult:
    scores: list[float]
    mode: str
    metadata: dict


def execute_vqc(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    mode: str = "exact",
    shots: int = 1024,
    seed: int = 42,
    ibm_config: IBMBackendConfig | None = None,
    service=None,
    local_backend=None,
) -> VQCExecutionResult:
    """Execute identical fixed VQC circuits through a selected backend."""
    if mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(f"Unsupported VQC execution mode: {mode}")
    if mode == "exact":
        return VQCExecutionResult(
            scores=_circuit_probabilities(angles, weights).tolist(),
            mode=mode,
            metadata={"backend": "numpy_statevector", "shots": None, "seed": seed},
        )
    if mode == "ibm_hardware":
        config = ibm_config or IBMBackendConfig(shots=shots)
        result = run_ibm_vqc(angles, weights, config, service=service)
        return VQCExecutionResult(result.scores, mode, result.metadata)
    return _execute_aer(
        angles,
        weights,
        shots=shots,
        seed=seed,
        noisy=mode == "noisy_simulator",
        backend=local_backend,
    )


def _execute_aer(angles, weights, *, shots, seed, noisy, backend):
    if shots <= 0:
        raise ValueError("shots must be positive for simulator execution")
    if noisy and backend is None:
        raise ValueError(
            "noisy_simulator requires an IBM/fake backend calibration snapshot"
        )
    try:
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import SamplerV2
    except ImportError as exc:
        raise RuntimeError(
            "Shot simulators require the 'quantum' dependencies."
        ) from exc
    if backend is None:
        backend = AerSimulator(seed_simulator=seed)
    aer = AerSimulator.from_backend(backend) if noisy else backend
    from market_qml.qml.ibm_backend import build_vqc_circuit

    circuits = [build_vqc_circuit(row, weights) for row in np.asarray(angles)]
    transpiled = generate_preset_pass_manager(backend=aer, optimization_level=1).run(
        circuits
    )
    result = SamplerV2(mode=aer).run(transpiled, shots=shots).result()
    scores = []
    for pub_result in result:
        counts = pub_result.data.c.get_counts()
        scores.append(counts.get("1", 0) / sum(counts.values()))
    return VQCExecutionResult(
        scores=scores,
        mode="noisy_simulator" if noisy else "shot_simulator",
        metadata={
            "backend": getattr(aer, "name", str(aer)),
            "shots": shots,
            "seed": seed,
        },
    )
