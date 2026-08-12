"""Bounded IBM Quantum Runtime execution for trained VQC circuits."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

IBM_CHANNEL = "ibm_quantum_platform"


@dataclass(frozen=True)
class IBMBackendConfig:
    backend_name: str | None = None
    shots: int = 1024
    max_circuits: int = 16
    max_total_shots: int = 16_384
    optimization_level: int = 1
    instance: str | None = None

    def validate(self, circuit_count: int) -> None:
        if self.shots <= 0 or self.max_circuits <= 0 or self.max_total_shots <= 0:
            raise ValueError("IBM Quantum execution limits must be positive.")
        if circuit_count <= 0 or circuit_count > self.max_circuits:
            raise ValueError(
                f"Circuit count {circuit_count} exceeds limit {self.max_circuits}."
            )
        if circuit_count * self.shots > self.max_total_shots:
            raise ValueError("Requested execution exceeds max_total_shots.")
        if self.optimization_level not in range(4):
            raise ValueError("optimization_level must be between 0 and 3.")


@dataclass(frozen=True)
class IBMExecutionResult:
    scores: list[float]
    job_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IBMSubmittedJob:
    job_id: str
    metadata: dict[str, Any]


def build_vqc_circuit(angles: np.ndarray, weights: np.ndarray):
    """Build the provider-native equivalent of the repository VQC circuit."""
    QuantumCircuit, _, _, _ = _qiskit_imports()
    values = np.asarray(angles, dtype=float)
    parameters = np.asarray(weights, dtype=float)
    if values.ndim != 1 or parameters.ndim != 2:
        raise ValueError("VQC angles must be 1D and weights must be 2D.")
    depth, n_qubits = parameters.shape
    if len(values) != n_qubits:
        raise ValueError("VQC angle and weight qubit counts must match.")
    circuit = QuantumCircuit(n_qubits, 1)
    for qubit, angle in enumerate(values):
        circuit.ry(float(angle), qubit)
    for layer in range(depth):
        for qubit in range(n_qubits):
            circuit.ry(float(parameters[layer, qubit]), qubit)
        for control in range(n_qubits):
            circuit.cx(control, (control + 1) % n_qubits)
    circuit.measure(0, 0)
    return circuit


def run_ibm_vqc(
    angles: np.ndarray,
    weights: np.ndarray,
    config: IBMBackendConfig,
    *,
    service=None,
) -> IBMExecutionResult:
    """Submit bounded VQC evaluation circuits and wait for Runtime results."""
    submitted = submit_ibm_vqc(angles, weights, config, service=service)
    return collect_ibm_vqc(
        submitted.job_id, config, service=service, metadata=submitted.metadata
    )


def submit_ibm_vqc(
    angles: np.ndarray,
    weights: np.ndarray,
    config: IBMBackendConfig,
    *,
    service=None,
) -> IBMSubmittedJob:
    """Submit bounded circuits and return immediately with a resumable job ID."""
    rows = np.asarray(angles, dtype=float)
    if rows.ndim != 2:
        raise ValueError("VQC evaluation angles must be two-dimensional.")
    config.validate(len(rows))
    _, generate_pass_manager, _, Sampler = _qiskit_imports()
    if service is None:
        service = _runtime_service(config)
    backend = (
        service.backend(config.backend_name)
        if config.backend_name
        else service.least_busy(
            min_num_qubits=np.asarray(weights).shape[1], operational=True
        )
    )
    circuits = [build_vqc_circuit(row, weights) for row in rows]
    pass_manager = generate_pass_manager(
        backend=backend, optimization_level=config.optimization_level
    )
    transpiled = pass_manager.run(circuits)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = config.shots
    submitted_at = time.perf_counter()
    job = sampler.run(transpiled, shots=config.shots)
    backend_name = getattr(backend, "name", str(backend))
    metadata = {
        "provider": "IBM Quantum",
        "channel": IBM_CHANNEL,
        "backend": backend_name,
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "shots": config.shots,
        "circuits": len(circuits),
        "total_shots": len(circuits) * config.shots,
        "job_id": job.job_id(),
        "submission_seconds": time.perf_counter() - submitted_at,
        "status": "SUBMITTED",
        "transpiled_depth": [circuit.depth() for circuit in transpiled],
        "gate_counts": [dict(circuit.count_ops()) for circuit in transpiled],
        "config": asdict(config),
    }
    return IBMSubmittedJob(job_id=job.job_id(), metadata=metadata)


def collect_ibm_vqc(
    job_id: str,
    config: IBMBackendConfig,
    *,
    service=None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> IBMExecutionResult:
    """Resume a Runtime job and collect scores and available provider metadata."""
    service = service or _runtime_service(config)
    job = service.job(job_id)
    started = time.perf_counter()
    primitive_result = job.result(timeout=timeout)
    scores = []
    for pub_result in primitive_result:
        counts = pub_result.data.c.get_counts()
        shots = sum(counts.values())
        scores.append(float(counts.get("1", 0) / shots))
    details = dict(metadata or {})
    details.update(
        {
            "job_id": job_id,
            "status": str(job.status()),
            "result_wait_seconds": time.perf_counter() - started,
            "job_metrics": _safe_job_value(job, "metrics", {}),
            "usage_estimation": _safe_job_value(job, "usage_estimation", {}),
            "calibration_metadata": _properties_metadata(
                _safe_job_value(job, "properties", None)
            ),
        }
    )
    return IBMExecutionResult(scores=scores, job_id=job_id, metadata=details)


def save_ibm_execution(
    result: IBMExecutionResult, output_dir: str | Path
) -> dict[str, Path]:
    """Persist resumable job identity, scores, and execution metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": directory / "ibm_execution_metadata.json",
        "scores": directory / "ibm_scores.json",
    }
    paths["metadata"].write_text(
        json.dumps(result.metadata, indent=2, default=str), encoding="utf-8"
    )
    paths["scores"].write_text(json.dumps(result.scores, indent=2), encoding="utf-8")
    return paths


def save_ibm_submission(result: IBMSubmittedJob, output_dir: str | Path) -> Path:
    """Persist the job ID immediately so interrupted runs remain resumable."""
    path = Path(output_dir) / "ibm_submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"job_id": result.job_id, "metadata": result.metadata},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _runtime_service(config: IBMBackendConfig):
    _, _, RuntimeService, _ = _qiskit_imports()
    token = os.getenv("IBM_QUANTUM_API_KEY")
    if not token:
        raise RuntimeError("IBM_QUANTUM_API_KEY is required for hardware runs.")
    return RuntimeService(
        channel=IBM_CHANNEL,
        token=token,
        instance=config.instance or os.getenv("IBM_QUANTUM_INSTANCE") or None,
    )


def _safe_job_value(job, name: str, default):
    try:
        value = getattr(job, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _properties_metadata(properties) -> dict[str, Any]:
    if properties is None:
        return {}
    if hasattr(properties, "to_dict"):
        return properties.to_dict()
    return {"repr": repr(properties)}


def _qiskit_imports():
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise RuntimeError(
            "IBM Quantum support requires the 'quantum' optional dependencies."
        ) from exc
    return QuantumCircuit, generate_preset_pass_manager, QiskitRuntimeService, SamplerV2
