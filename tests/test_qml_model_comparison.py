import numpy as np
import pandas as pd

import market_qml.qml.comparison as comparison
from market_qml.qml.comparison import (
    DEFAULT_FEATURE_SELECTIONS,
    ComparisonConfig,
    aggregate_split_metrics,
    run_model_comparison,
    save_comparison_result,
)
from scripts.compare_qml_models import _write_hardware_qualification


def _comparison_data():
    columns = sorted(
        {column for values in DEFAULT_FEATURE_SELECTIONS.values() for column in values}
    )
    rows = []
    for split_id in [0, 1]:
        for role, start in [("train", "2020-01-01"), ("validation", "2020-03-01")]:
            for index in range(16):
                target = index % 2
                row = {
                    "symbol": f"S{index:02d}",
                    "date": pd.Timestamp(start) + pd.Timedelta(days=index),
                    "split_id": split_id,
                    "sample_role": role,
                    "target": target,
                    "forward_return_5d": 0.02 if target else -0.01,
                    "forward_excess_return_5d": 0.01 if target else -0.02,
                }
                row.update(
                    {
                        column: np.sin(index + offset)
                        for offset, column in enumerate(columns)
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_comparison_uses_identical_rows_and_training_only_qsvm_selection():
    result = run_model_comparison(
        _comparison_data(),
        ComparisonConfig(
            train_rows=16,
            validation_rows=16,
            vqc_iterations=1,
            qcnn_iterations=1,
            qsvm_c_values=(1.0,),
            qsvm_repetitions=(1,),
            interaction_scales=(0.0,),
            vqc_ansatz_depths=(1,),
            vqc_learning_rates=(0.1,),
            vqc_optimizers=("spsa",),
            qcnn_learning_rates=(0.05,),
            qcnn_initialization_scales=(0.1,),
            feature_selection_names=("broad_market",),
            bootstrap_iterations=20,
            inner_purge_days=0,
        ),
    )
    assert set(result.predictions.model_name) == {
        "vqc",
        "vqc_stable_rank",
        "qcnn",
        "qsvm",
        "qsvm_tuned",
        "linear_svm",
        "rbf_svm",
        "logistic_regression",
        "gradient_boosting",
    }
    counts = result.predictions.groupby(["split_id", "model_name"]).size()
    assert counts.nunique() == 1
    key_hashes = (
        result.predictions.assign(
            key=lambda frame: frame.symbol.astype(str) + frame.date.astype(str)
        )
        .groupby(["split_id", "model_name"])
        .key.apply(lambda values: tuple(sorted(values)))
    )
    assert key_hashes.groupby(level=0).nunique().eq(1).all()
    assert (
        result.qsvm_tuning_trials.inner_train_end
        < result.qsvm_tuning_trials.inner_validation_start
    ).all()
    assert (
        result.vqc_tuning_trials.inner_train_end
        < result.vqc_tuning_trials.inner_validation_start
    ).all()
    assert (
        result.qcnn_tuning_trials.inner_train_end
        < result.qcnn_tuning_trials.inner_validation_start
    ).all()
    assert result.qsvm_tuning_trials.inner_fold_id.nunique() >= 2
    assert result.vqc_tuning_trials.inner_fold_id.nunique() >= 2
    assert result.qcnn_tuning_trials.inner_fold_id.nunique() >= 2
    assert set(result.aggregate_metrics.columns) >= {
        "mean",
        "median",
        "std",
        "ci_lower",
        "ci_upper",
    }
    assert set(result.ranking_metrics.scope) == {"date", "split", "overall"}
    assert set(result.portfolio_metrics.scope) == {"split", "overall"}
    assert set(result.portfolio_metrics.model_name) == set(
        result.predictions.model_name
    )
    assert set(result.paired_comparisons.columns) >= {
        "mean_difference",
        "ci_lower",
        "ci_upper",
        "permutation_p_value",
        "holm_adjusted_p_value",
        "practically_meaningful",
        "decision",
    }
    assert not result.date_block_metrics.empty


def test_aggregate_and_save_outputs(tmp_path):
    metrics = pd.DataFrame(
        {
            "model_name": ["a", "a", "b", "b"],
            "split_id": [0, 1, 0, 1],
            "accuracy": [0.5, 0.7, 0.4, 0.6],
        }
    )
    aggregate = aggregate_split_metrics(metrics, bootstrap_iterations=20)
    assert aggregate.loc[aggregate.model_name == "a", "mean"].iloc[0] == 0.6

    result = run_model_comparison(
        _comparison_data().query("split_id == 0"),
        ComparisonConfig(
            train_rows=16,
            validation_rows=16,
            vqc_iterations=1,
            qcnn_iterations=1,
            qsvm_c_values=(1.0,),
            qsvm_repetitions=(1,),
            feature_selection_names=("broad_market",),
            interaction_scales=(0.0,),
            vqc_ansatz_depths=(1,),
            vqc_learning_rates=(0.1,),
            vqc_optimizers=("spsa",),
            qcnn_learning_rates=(0.05,),
            qcnn_initialization_scales=(0.1,),
            bootstrap_iterations=10,
            inner_purge_days=0,
        ),
    )
    paths = save_comparison_result(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert "Decision:" in paths["report"].read_text()
    assert {
        "ranking_metrics",
        "portfolio_returns",
        "portfolio_metrics",
        "vqc_tuning_trials",
        "vqc_selected_configs",
        "qcnn_tuning_trials",
        "qcnn_selected_configs",
        "paired_comparisons",
        "date_block_metrics",
    } <= set(paths)


def test_qsvm_tuning_reuses_feature_map_states_across_c_values(monkeypatch):
    calls = 0
    original = comparison.QuantumKernelFeatureMap.transform

    def counted_transform(self, dataset):
        nonlocal calls
        calls += 1
        return original(self, dataset)

    monkeypatch.setattr(
        comparison.QuantumKernelFeatureMap, "transform", counted_transform
    )
    data = _comparison_data().query("split_id == 0")
    config = ComparisonConfig(
        train_rows=16,
        validation_rows=16,
        qsvm_c_values=(0.1, 1.0, 10.0),
        qsvm_repetitions=(1,),
        interaction_scales=(0.0,),
        feature_selection_names=("broad_market",),
        inner_purge_days=0,
    )
    sampled = comparison._sample_split(data, 0, config)
    folds = comparison._prepared_inner_folds(sampled, 0, config)

    _, trials = comparison._select_qsvm(sampled, 0, config, prepared_folds=folds)

    assert len(trials) == len(folds["broad_market"]) * 3
    assert calls == len(folds["broad_market"]) * 2


def test_hardware_qualification_requires_stable_vqc_to_beat_controls(tmp_path):
    metrics = pd.DataFrame(
        {
            "scope": ["split"] * 6,
            "model_name": ["vqc_stable_rank"] * 3 + ["linear_svm"] * 3,
            "rank_information_coefficient": [0.1, 0.2, 0.1, 0.01, 0.02, 0.03],
        }
    )
    path = _write_hardware_qualification(metrics, tmp_path)

    import json

    assert json.loads(path.read_text())["qualified_for_hardware"] is True
