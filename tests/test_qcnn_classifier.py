import pandas as pd
import pytest

from market_qml.models.predictions import REQUIRED_PREDICTION_COLUMNS
from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.qcnn import MODEL_NAME, save_qcnn_result, train_qcnn


def _sample() -> pd.DataFrame:
    rows = []
    for role, targets in {
        "train": [0, 1, 0, 1, 0, 1],
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
            for component in range(8):
                row[f"pca_{component:02d}"] = (
                    target * 0.6 + index * 0.02 + component * 0.05
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_qcnn_trains_and_outputs_predictions_loss_and_metrics():
    result = train_qcnn(
        build_qml_train_validation(_sample(), split_id=0),
        max_iter=3,
        batch_size=4,
        random_state=7,
    )

    assert result.config.model_name == MODEL_NAME
    assert result.config.params["n_qubits"] == 8
    assert result.config.params["readout_qubits"] == [0, 4]
    assert result.model.weights_.shape == (30,)
    assert list(result.predictions.columns) == REQUIRED_PREDICTION_COLUMNS
    assert result.predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert result.predictions["y_score"].between(0, 1).all()
    assert result.training_loss["iteration"].tolist() == [1, 2, 3]
    assert result.training_loss["loss"].notna().all()
    assert result.training_metrics["sample_role"].tolist() == ["train"]
    assert result.validation_metrics["sample_role"].tolist() == ["validation"]
    assert result.validation_metrics["rows"].tolist() == [4]


def test_qcnn_training_is_reproducible():
    data = build_qml_train_validation(_sample(), split_id=0)

    first = train_qcnn(data, max_iter=2, random_state=19)
    second = train_qcnn(data, max_iter=2, random_state=19)

    assert first.model.weights_ == pytest.approx(second.model.weights_)
    assert first.training_loss["loss"].to_numpy() == pytest.approx(
        second.training_loss["loss"].to_numpy()
    )
    assert first.predictions["y_score"].to_numpy() == pytest.approx(
        second.predictions["y_score"].to_numpy()
    )


def test_qcnn_outputs_can_be_saved(tmp_path):
    result = train_qcnn(
        build_qml_train_validation(_sample(), split_id=0),
        max_iter=2,
    )

    paths = save_qcnn_result(result, output_dir=tmp_path)

    assert set(paths) == {
        "model",
        "predictions",
        "training_loss",
        "training_metrics",
        "validation_metrics",
    }
    assert all(path.exists() for path in paths.values())
    assert pd.read_parquet(paths["predictions"])["model_name"].tolist() == [
        MODEL_NAME
    ] * 4


def test_qcnn_rejects_single_class_training_targets():
    sample = _sample()
    sample["target"] = 0
    with pytest.raises(ValueError, match="two classes"):
        train_qcnn(
            build_qml_train_validation(sample, split_id=0),
            max_iter=1,
        )
