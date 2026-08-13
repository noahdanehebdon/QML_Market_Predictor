import numpy as np
import pandas as pd
import pytest

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.vqc import (
    MODEL_NAME,
    _circuit_probabilities,
    save_predictions,
    save_training_loss,
    save_validation_metrics,
    train_vqc,
)


def _qml_sample() -> pd.DataFrame:
    rows = []
    labels = {
        "train": [0, 1, 0, 1, 0, 1],
        "validation": [0, 1, 0, 1],
    }
    for role, targets in labels.items():
        for row_index, target in enumerate(targets):
            row = {
                "symbol": f"SYM{row_index}",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=row_index),
                "split_id": 0,
                "sample_role": role,
                "target": target,
                "forward_return_5d": 0.02 if target else -0.01,
                "forward_excess_return_5d": 0.01 if target else -0.02,
            }
            for component_index in range(8):
                row[f"pca_{component_index:02d}"] = (
                    float(target) + component_index * 0.1 + row_index * 0.01
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_vqc_outputs_standard_predictions_loss_and_metrics():
    data = build_qml_train_validation(_qml_sample(), split_id=0)

    result = train_vqc(
        data,
        max_iter=12,
        learning_rate=0.05,
        random_state=7,
    )

    assert result.config.model_name == MODEL_NAME
    assert result.config.seed == 7
    assert result.config.params["n_qubits"] == 8
    assert result.config.params["ansatz"] == "ry_ring_cnot"
    assert result.config.params["simulator"] == "numpy_statevector"
    assert result.config.params["optimizer"] == "spsa"
    assert result.model.weights_.shape == (1, 8)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["split_id"].unique().tolist() == [0]
    assert result.predictions["y_score"].between(0, 1).all()
    assert result.training_loss["iteration"].tolist() == list(range(1, 13))
    assert result.training_loss["loss"].notna().all()
    assert result.validation_metrics["model_name"].tolist() == [MODEL_NAME]
    assert set(result.validation_metrics.columns) == {
        "model_name",
        "split_id",
        "log_loss",
        "accuracy",
        "brier_score",
    }


def test_vqc_outputs_can_be_saved(tmp_path):
    result = train_vqc(
        build_qml_train_validation(_qml_sample(), split_id=0),
        max_iter=3,
    )
    prediction_path = tmp_path / "predictions.parquet"
    loss_path = tmp_path / "loss.parquet"
    metrics_path = tmp_path / "metrics.parquet"

    save_predictions(result.predictions, prediction_path)
    save_training_loss(result.training_loss, loss_path)
    save_validation_metrics(result.validation_metrics, metrics_path)

    assert pd.read_parquet(prediction_path)["model_name"].unique().tolist() == [
        MODEL_NAME
    ]
    assert pd.read_parquet(loss_path)["iteration"].tolist() == [1, 2, 3]
    assert pd.read_parquet(metrics_path)["split_id"].tolist() == [0]


def test_train_vqc_rejects_non_binary_targets():
    sample = _qml_sample()
    sample["target"] = 0.5

    with pytest.raises(ValueError, match="binary targets"):
        train_vqc(build_qml_train_validation(sample, split_id=0))


def test_vqc_statevector_ansatz_entangles_feature_qubits_with_readout():
    weights = np.zeros((1, 2))
    probabilities = _circuit_probabilities(
        np.array(
            [
                [0.0, 0.0],
                [0.0, np.pi],
            ]
        ),
        weights,
    )

    assert probabilities == pytest.approx([0.0, 1.0], abs=1e-12)


def test_vqc_training_is_reproducible_for_the_same_seed():
    data = build_qml_train_validation(_qml_sample(), split_id=0)

    first = train_vqc(data, max_iter=4, random_state=17)
    second = train_vqc(data, max_iter=4, random_state=17)

    assert first.model.weights_ == pytest.approx(second.model.weights_)
    assert first.training_loss["loss"].to_numpy() == pytest.approx(
        second.training_loss["loss"].to_numpy()
    )
    assert first.predictions["y_score"].to_numpy() == pytest.approx(
        second.predictions["y_score"].to_numpy()
    )


def test_vqc_uses_same_date_continuous_ranking_targets_when_available():
    sample = _qml_sample()
    sample["date"] = (
        sample.groupby("sample_role")
        .cumcount()
        .floordiv(2)
        .map(lambda index: pd.Timestamp("2024-01-01") + pd.Timedelta(days=index))
    )
    sample["ranking_target"] = sample["forward_excess_return_5d"]
    data = build_qml_train_validation(sample, split_id=0)

    result = train_vqc(data, max_iter=3, random_state=5)

    assert len(result.training_loss) == 3
    assert result.training_loss["loss"].notna().all()
