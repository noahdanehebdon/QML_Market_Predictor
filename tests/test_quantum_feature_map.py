import numpy as np
import pandas as pd
import pytest

from market_qml.qml.feature_map import (
    QuantumFeatureMapConfig,
    QuantumKernelFeatureMap,
    fidelity_kernel,
    save_feature_map_split,
)
from market_qml.qml.interface import build_qml_train_validation


def _sample() -> pd.DataFrame:
    rows = []
    for role, count in [("train", 4), ("validation", 2)]:
        for row_index in range(count):
            row = {
                "symbol": f"SYM{row_index}",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=row_index),
                "split_id": 0,
                "sample_role": role,
                "target": row_index % 2,
            }
            for component in range(3):
                row[f"pca_{component:02d}"] = row_index * 0.2 + component * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_feature_map_outputs_normalized_states_and_circuit_description():
    data = build_qml_train_validation(_sample(), split_id=0)
    feature_map = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3, repetitions=2)
    )

    result = feature_map.transform(data.train)

    assert result.states.shape == (4, 8)
    assert np.sum(np.abs(result.states) ** 2, axis=1) == pytest.approx(
        np.ones(4)
    )
    assert result.angles.shape == (4, 3)
    assert result.metadata["symbol"].tolist() == ["SYM0", "SYM1", "SYM2", "SYM3"]
    assert result.config.backend == "numpy_statevector"
    assert len(result.operations) == 12
    assert result.operations["gate"].tolist().count("ry") == 6
    assert result.operations["gate"].tolist().count("cz") == 6


def test_feature_map_applies_identical_circuit_to_train_and_validation():
    data = build_qml_train_validation(_sample(), split_id=0)
    result = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3)
    ).transform_train_validation(data)

    assert result.split_id == 0
    assert result.train.states.shape == (4, 8)
    assert result.validation.states.shape == (2, 8)
    assert result.train.operations.equals(result.validation.operations)


def test_feature_map_adds_selected_feature_interaction_reuploading():
    data = build_qml_train_validation(_sample(), split_id=0)
    baseline = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3, interaction_scale=0.0)
    ).transform(data.train)
    redesigned = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3, interaction_scale=0.5)
    ).transform(data.train)

    assert redesigned.operations["gate"].tolist().count("ry_interaction") == 6
    assert not np.allclose(baseline.states, redesigned.states)


def test_fidelity_kernel_is_symmetric_bounded_and_has_unit_diagonal():
    data = build_qml_train_validation(_sample(), split_id=0)
    feature_map = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3)
    )

    kernel = feature_map.kernel(data.train)
    cross_kernel = feature_map.kernel(data.validation, data.train)

    assert kernel.shape == (4, 4)
    assert kernel == pytest.approx(kernel.T, abs=1e-12)
    assert np.diag(kernel) == pytest.approx(np.ones(4), abs=1e-12)
    assert ((kernel >= 0.0) & (kernel <= 1.0)).all()
    assert cross_kernel.shape == (2, 4)


def test_fidelity_kernel_rejects_invalid_or_mismatched_states():
    with pytest.raises(ValueError, match="non-normalized"):
        fidelity_kernel(np.array([[2.0, 0.0]]), np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="widths must match"):
        fidelity_kernel(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0, 0.0]]))


def test_feature_map_split_outputs_can_be_saved_for_kernel_estimation(tmp_path):
    data = build_qml_train_validation(_sample(), split_id=0)
    result = QuantumKernelFeatureMap(
        QuantumFeatureMapConfig(n_qubits=3)
    ).transform_train_validation(data)

    paths = save_feature_map_split(result, output_dir=tmp_path)
    states = np.load(paths["states"])

    assert set(paths) == {
        "states",
        "train_metadata",
        "validation_metadata",
        "operations",
    }
    assert states["train_states"].shape == (4, 8)
    assert states["validation_states"].shape == (2, 8)
    assert len(pd.read_parquet(paths["operations"])) == 12


@pytest.mark.parametrize(
    "config,message",
    [
        (QuantumFeatureMapConfig(n_qubits=1), "at least 2"),
        (QuantumFeatureMapConfig(repetitions=0), "repetitions"),
        (QuantumFeatureMapConfig(backend="other"), "Unsupported"),
        (QuantumFeatureMapConfig(entanglement="full"), "ring"),
        (QuantumFeatureMapConfig(interaction_scale=-0.1), "non-negative"),
    ],
)
def test_feature_map_rejects_unsupported_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        QuantumKernelFeatureMap(config)
