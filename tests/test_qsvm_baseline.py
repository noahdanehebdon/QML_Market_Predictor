import numpy as np
import pandas as pd
import pytest

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qsvm import MODEL_NAME, save_qsvm_result, train_qsvm


def _sample() -> pd.DataFrame:
    rows = []
    for role, targets in {
        "train": [0, 1, 0, 1, 0, 1, 0, 1],
        "validation": [0, 1, 0, 1],
    }.items():
        for index, target in enumerate(targets):
            row = {
                "symbol": f"SYM{index}",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                "split_id": 0,
                "sample_role": role,
                "target": target,
                "forward_return_5d": 0.02 if target else -0.01,
                "forward_excess_return_5d": 0.01 if target else -0.02,
            }
            for component in range(3):
                row[f"pca_{component:02d}"] = (
                    target * 0.7 + index * 0.03 + component * 0.1
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_qsvm_builds_expected_kernels_and_standard_predictions():
    result = train_qsvm(
        build_qml_train_validation(_sample(), split_id=0),
        n_qubits=3,
        repetitions=2,
        random_state=7,
    )

    assert result.config.model_name == MODEL_NAME
    assert result.config.params["kernel"] == "quantum_state_fidelity"
    assert result.train_kernel.shape == (8, 8)
    assert result.validation_kernel.shape == (4, 8)
    assert result.train_kernel == pytest.approx(result.train_kernel.T, abs=1e-12)
    assert np.diag(result.train_kernel) == pytest.approx(np.ones(8), abs=1e-12)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["y_score"].between(0, 1).all()
    assert result.kernel_diagnostics[["rows", "columns"]].to_dict("records") == [
        {"rows": 8, "columns": 8},
        {"rows": 4, "columns": 8},
    ]


def test_qsvm_artifacts_can_be_saved(tmp_path):
    result = train_qsvm(
        build_qml_train_validation(_sample(), split_id=0),
        n_qubits=3,
    )

    paths = save_qsvm_result(
        result,
        model_path=tmp_path / "model.pkl",
        prediction_path=tmp_path / "predictions.parquet",
        diagnostics_path=tmp_path / "diagnostics.parquet",
        kernel_path=tmp_path / "kernels.npz",
    )
    kernels = np.load(paths["kernel_matrices"])

    assert all(path.exists() for path in paths.values())
    assert kernels["train_kernel"].shape == (8, 8)
    assert kernels["validation_kernel"].shape == (4, 8)
    assert pd.read_parquet(paths["predictions"])["model_name"].tolist() == [MODEL_NAME] * 4


def test_qsvm_rejects_invalid_targets_and_regularization():
    sample = _sample()
    sample["target"] = 0
    data = build_qml_train_validation(sample, split_id=0)
    with pytest.raises(ValueError, match="two classes"):
        train_qsvm(data, n_qubits=3)

    data = build_qml_train_validation(_sample(), split_id=0)
    with pytest.raises(ValueError, match="C must be positive"):
        train_qsvm(data, n_qubits=3, C=0)
