import numpy as np
import pytest

from market_qml.qml.execution_backend import execute_vqc


def test_exact_backend_matches_standard_vqc_probability():
    result = execute_vqc(
        np.array([[0.0, 0.0], [0.0, np.pi]]),
        np.zeros((1, 2)),
    )
    assert result.mode == "exact"
    assert result.scores == pytest.approx([0.0, 1.0])


def test_backend_rejects_unknown_mode_and_unbounded_noisy_backend():
    with pytest.raises(ValueError, match="Unsupported"):
        execute_vqc(np.zeros((1, 2)), np.zeros((1, 2)), mode="unknown")
    with pytest.raises(ValueError, match="calibration snapshot"):
        execute_vqc(np.zeros((1, 2)), np.zeros((1, 2)), mode="noisy_simulator")
