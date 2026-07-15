import numpy as np
import pytest

from market_qml.qml.simulator import (
    apply_cnot,
    apply_cz,
    apply_ry,
    expectation_z,
    zero_state,
)


def test_statevector_gates_preserve_norm_and_apply_expected_basis_changes():
    state = zero_state(1, 2)
    state = apply_ry(state, np.pi, 0)
    state = apply_cnot(state, 0, 1)

    assert state[0] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-12)
    assert np.sum(np.abs(state) ** 2) == pytest.approx(1.0)

    phased = apply_cz(state, 0, 1)
    assert phased[0] == pytest.approx([0.0, 0.0, 0.0, -1.0], abs=1e-12)
    assert expectation_z(phased, 0) == pytest.approx([-1.0])
    assert expectation_z(phased, 1) == pytest.approx([-1.0])


def test_statevector_backend_validates_gate_inputs():
    with pytest.raises(ValueError, match="positive"):
        zero_state(1, 0)
    with pytest.raises(ValueError, match="distinct"):
        apply_cnot(zero_state(1, 2), 0, 0)
    with pytest.raises(ValueError, match="one value per state"):
        apply_ry(zero_state(2, 2), np.array([0.1]), 0)
