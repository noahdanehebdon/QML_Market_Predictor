import json

import pandas as pd
import pytest

from market_qml.qml.interface import build_qml_train_validation
from market_qml.qml.tuning import (
    render_vqc_tuning_report,
    save_vqc_tuning_result,
    tune_vqc,
)
from market_qml.qml.vqc import train_vqc


def _qml_sample() -> pd.DataFrame:
    rows = []
    targets_by_role = {
        "train": [0, 1, 0, 1, 0, 1],
        "validation": [0, 1, 0, 1],
    }
    for role, targets in targets_by_role.items():
        for index, target in enumerate(targets):
            rows.append(
                {
                    "symbol": f"SYM{index}",
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index),
                    "split_id": 0,
                    "sample_role": role,
                    "target": target,
                    "forward_return_5d": 0.01 if target else -0.01,
                    "forward_excess_return_5d": 0.02 if target else -0.02,
                    "pca_00": float(target) + index * 0.02,
                    "pca_01": float(1 - target) - index * 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_tune_vqc_compares_depth_rate_and_optimizer_and_selects_best():
    sample = _qml_sample().drop(
        columns=["forward_return_5d", "forward_excess_return_5d"]
    )
    data = build_qml_train_validation(sample, split_id=0)

    result = tune_vqc(
        data,
        ansatz_depths=[1, 2],
        learning_rates=[0.05],
        optimizers=["spsa", "finite_difference"],
        max_iter=2,
        n_qubits=2,
        batch_size=4,
        random_state=7,
    )

    assert len(result.results) == 4
    assert result.results["rank"].tolist() == [1, 2, 3, 4]
    assert set(result.results["ansatz_depth"]) == {1, 2}
    assert set(result.results["optimizer"]) == {"spsa", "finite_difference"}
    assert result.results["train_log_loss"].notna().all()
    assert result.results["validation_log_loss"].notna().all()
    assert result.results["overfitting_flag"].dtype == bool
    assert len(result.loss_history) == 8
    assert result.best_config["config_id"] == result.results.iloc[0]["config_id"]


def test_tuning_outputs_document_best_config_and_loss_history(tmp_path):
    result = tune_vqc(
        build_qml_train_validation(_qml_sample(), split_id=0),
        ansatz_depths=[1],
        learning_rates=[0.1],
        optimizers=["spsa"],
        max_iter=2,
        n_qubits=2,
    )

    paths = save_vqc_tuning_result(result, output_dir=tmp_path)
    saved_config = json.loads(paths["best_config"].read_text(encoding="utf-8"))
    report = paths["report"].read_text(encoding="utf-8")

    assert set(paths) == {"results", "loss_history", "best_config", "report"}
    assert all(path.exists() for path in paths.values())
    assert saved_config == result.best_config
    assert "Best configuration" in report
    assert result.best_config["config_id"] in report
    assert render_vqc_tuning_report(result) == report


def test_vqc_supports_finite_difference_and_rejects_unknown_optimizer():
    data = build_qml_train_validation(_qml_sample(), split_id=0)

    result = train_vqc(
        data,
        n_qubits=2,
        optimizer="finite_difference",
        max_iter=2,
    )

    assert result.config.params["optimizer"] == "finite_difference"
    assert result.training_loss["loss"].notna().all()
    with pytest.raises(ValueError, match="optimizer must be one of"):
        train_vqc(data, n_qubits=2, optimizer="unknown", max_iter=1)


@pytest.mark.parametrize(
    ("depths", "rates", "optimizers", "message"),
    [
        ([], [0.1], ["spsa"], "ansatz_depths"),
        ([1], [0.0], ["spsa"], "learning_rates"),
        ([1], [0.1], [], "optimizers"),
    ],
)
def test_tune_vqc_rejects_invalid_grids(depths, rates, optimizers, message):
    with pytest.raises(ValueError, match=message):
        tune_vqc(
            build_qml_train_validation(_qml_sample(), split_id=0),
            ansatz_depths=depths,
            learning_rates=rates,
            optimizers=optimizers,
            max_iter=1,
            n_qubits=2,
        )
