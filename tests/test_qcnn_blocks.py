import numpy as np
import pytest

from market_qml.qml.qcnn_blocks import (
    build_qcnn_architecture,
    execute_qcnn_architecture,
    initialize_qcnn_parameters,
    pooling_block,
    two_qubit_convolution,
)
from market_qml.qml.simulator import apply_ry, zero_state


def test_qcnn_architecture_has_expected_active_flow_and_parameter_shapes():
    architecture = build_qcnn_architecture()

    assert architecture.n_qubits == 8
    assert architecture.active_qubit_flow == (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 2, 4, 6),
        (0, 4),
    )
    assert architecture.parameter_count == 30
    assert architecture.block_parameter_shapes == {
        "convolution": (4,),
        "pooling": (1,),
        "complete_circuit": (30,),
    }
    assert len(architecture.operations) == 48
    assert architecture.operations["operation_index"].tolist() == list(range(48))
    assert architecture.operations["parameter_index"].dropna().astype(
        int
    ).tolist() == list(range(30))


def test_qcnn_parameter_initialization_is_bounded_and_reproducible():
    architecture = build_qcnn_architecture()

    first = initialize_qcnn_parameters(architecture, random_state=7, scale=0.2)
    second = initialize_qcnn_parameters(architecture, random_state=7, scale=0.2)

    assert first == pytest.approx(second)
    assert first.shape == (30,)
    assert np.abs(first).max() <= 0.2


def test_convolution_and_pooling_blocks_preserve_state_norm():
    state = zero_state(2, 8)
    state = apply_ry(state, np.array([0.2, -0.4]), 0)
    state = apply_ry(state, np.array([0.5, 0.1]), 1)

    convolved = two_qubit_convolution(
        state,
        0,
        1,
        np.array([0.1, -0.2, 0.3, -0.4]),
    )
    pooled = pooling_block(convolved, 1, 0, np.array([0.25]))

    assert np.sum(np.abs(convolved) ** 2, axis=1) == pytest.approx(np.ones(2))
    assert np.sum(np.abs(pooled) ** 2, axis=1) == pytest.approx(np.ones(2))
    assert not np.allclose(pooled, state)


def test_constructed_qcnn_executes_without_training_and_preserves_norm():
    architecture = build_qcnn_architecture()
    parameters = initialize_qcnn_parameters(architecture, random_state=11)
    state = zero_state(3, 8)
    state = apply_ry(state, np.array([0.1, 0.2, 0.3]), 3)

    result = execute_qcnn_architecture(state, architecture, parameters)

    assert result.shape == (3, 256)
    assert np.sum(np.abs(result) ** 2, axis=1) == pytest.approx(np.ones(3))


def test_qcnn_blocks_validate_qubit_count_and_parameter_shapes():
    with pytest.raises(ValueError, match="requires 8 qubits"):
        build_qcnn_architecture(4)

    state = zero_state(1, 8)
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        two_qubit_convolution(state, 0, 1, np.zeros(3))
    with pytest.raises(ValueError, match=r"shape \(1,\)"):
        pooling_block(state, 1, 0, np.zeros(2))
    with pytest.raises(ValueError, match="scale must be positive"):
        initialize_qcnn_parameters(build_qcnn_architecture(), scale=0)
