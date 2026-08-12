import json

import numpy as np
import pytest

import market_qml.qml.ibm_backend as ibm


class FakeCircuit:
    def __init__(self, qubits, classical_bits):
        self.qubits = qubits
        self.operations = []

    def ry(self, angle, qubit):
        self.operations.append(("ry", angle, qubit))

    def cx(self, control, target):
        self.operations.append(("cx", control, target))

    def measure(self, qubit, bit):
        self.operations.append(("measure", qubit, bit))

    def depth(self):
        return len(self.operations)

    def count_ops(self):
        return {"ry": 4, "cx": 2, "measure": 1}


def fake_imports():
    class PassManager:
        def run(self, circuits):
            return circuits

    class Sampler:
        def __init__(self, mode):
            self.options = type("Options", (), {})()

        def run(self, circuits, shots):
            result = []
            for index, _ in enumerate(circuits):
                ones = shots // 4 if index == 0 else shots // 2
                counts = {"0": shots - ones, "1": ones}
                register = type(
                    "Register", (), {"get_counts": lambda self, c=counts: c}
                )()
                result.append(
                    type("Pub", (), {"data": type("Data", (), {"c": register})()})()
                )
            return type(
                "Job",
                (),
                {
                    "job_id": lambda self: "job-123",
                    "result": lambda self, timeout=None: result,
                    "status": lambda self: "DONE",
                    "metrics": lambda self: {"usage": {"quantum_seconds": 1}},
                    "properties": lambda self: None,
                },
            )()

    return FakeCircuit, lambda **kwargs: PassManager(), object, Sampler


def test_config_enforces_circuit_and_shot_limits():
    config = ibm.IBMBackendConfig(shots=100, max_circuits=2, max_total_shots=150)
    with pytest.raises(ValueError, match="max_total_shots"):
        config.validate(2)
    with pytest.raises(ValueError, match="Circuit count"):
        config.validate(3)


def test_build_vqc_circuit_preserves_gate_order(monkeypatch):
    monkeypatch.setattr(ibm, "_qiskit_imports", fake_imports)
    circuit = ibm.build_vqc_circuit(np.array([0.1, 0.2]), np.array([[0.3, 0.4]]))
    assert [operation[0] for operation in circuit.operations] == [
        "ry",
        "ry",
        "ry",
        "ry",
        "cx",
        "cx",
        "measure",
    ]
    assert circuit.operations[-1] == ("measure", 0, 0)


def test_hardware_execution_uses_mocked_provider_and_persists_job(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ibm, "_qiskit_imports", fake_imports)
    backend = type("Backend", (), {"name": "fake_backend"})()
    completed_job = fake_imports()[3](backend).run([object(), object()], 100)
    service = type(
        "Service",
        (),
        {
            "backend": lambda self, name: backend,
            "job": lambda self, job_id: completed_job,
        },
    )()
    submitted = ibm.submit_ibm_vqc(
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5, 0.6]]),
        ibm.IBMBackendConfig(
            backend_name="fake_backend",
            shots=100,
            max_circuits=2,
            max_total_shots=200,
        ),
        service=service,
    )
    submission_path = ibm.save_ibm_submission(submitted, tmp_path)
    assert json.loads(submission_path.read_text())["job_id"] == "job-123"
    result = ibm.collect_ibm_vqc(
        submitted.job_id,
        ibm.IBMBackendConfig(shots=100),
        service=service,
        metadata=submitted.metadata,
    )
    assert result.scores == [0.25, 0.5]
    assert result.job_id == "job-123"
    paths = ibm.save_ibm_execution(result, tmp_path)
    assert json.loads(paths["metadata"].read_text())["job_id"] == "job-123"
